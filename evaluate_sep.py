import os
import torch
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from graph import Graph
from model import GenerativeQueryNetwork, DeformationGNet
from pool import FeaturePooling
from metrics import chamfer_loss, loss_function, f1_score
from data import CustomDatasetFolder, get_random_viewpoint, vgg_normalize

class Annealer(object):
    def __init__(self, init, delta, steps):
        self.init = init
        self.delta = delta
        self.steps = steps
        self.s = 0
        self.data = self.__repr__()
        self.recent = init

    def __repr__(self):
        return {"init": self.init, "delta": self.delta, "steps": self.steps, "s": self.s}

    def __iter__(self):
        return self

    def __next__(self):
        self.s += 1
        value = max(self.delta + (self.init - self.delta) * (1 - self.s / self.steps), self.delta)
        self.recent = value
        return value

    def state_dict(self):
        return {'init': self.init, 'delta': self.delta, 'steps': self.steps, 's': self.s, 'data': self.data, 'recent': self.recent}

    def load_state_dict(self, state_dict):
        self.init = state_dict['init']
        self.delta = state_dict['delta']
        self.steps = state_dict['steps']
        self.s = state_dict['s']
        self.data = state_dict['data']
        self.recent = state_dict['recent']

# Args
parser = argparse.ArgumentParser(description='Pixel2Mesh evaluating script')
parser.add_argument('--data', type=str, metavar='D',
                    help="folder where data is located.")
parser.add_argument('--log_step', type=int, default=100, metavar='L',
                    help='how many batches to wait before logging evaluation status')
parser.add_argument('--output', type=str, default=None, metavar='G',
                    help='if not None, generate meshes to this folder')
parser.add_argument('--show_img', type=bool, default=False, metavar='S',
                    help='whether or not to show the images')
parser.add_argument('--load_gqn', type=str, metavar='M',
                    help='GQN model file to load for evaluating.')
parser.add_argument('--load_gcn', type=str, metavar='M',
                    help='GCN model file to load for evaluating.')
args = parser.parse_args()

# Model
nIms = 20
nQuery = 5
model_gqn = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8)
model_gcn = DeformationGNet(nIms)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gqn_state_dict = torch.load(args.load_gqn, map_location=device)
model_gqn.load_state_dict(gqn_state_dict["model"])
gcn_state_dict = torch.load(args.load_gcn, map_location=device)
model_gcn.load_state_dict(gcn_state_dict)

model_gqn.eval()
model_gcn.eval()

# Cuda
use_cuda = torch.cuda.is_available()
if use_cuda:
    model_gqn.cuda()
    model_gcn.cuda()
    print('Using GPU')
else:
    print('Using CPU')

# Graph
graph = Graph("./ellipsoid/init_info.pickle")

# Data Loader
folder = CustomDatasetFolder(args.data, extensions = ["png"], dimension=nIms, print_ref=False)
test_loader = torch.utils.data.DataLoader(folder, batch_size=1, shuffle=True)

tot_loss_norm = 0
tot_loss_unorm = 0
tot_f1_1 = 0
tot_f1_2 = 0
tau = 1e-4
log_step = args.log_step
show_img = args.show_img

sigma_scheme = Annealer(2.0, 0.7, 80000)

for n, data in enumerate(test_loader):
    ims, t_ims, viewpoints, gt_points, gt_normals = data
    t_ims = np.transpose(t_ims, (1, 0, 2, 3, 4))
    distance = np.linalg.norm(viewpoints[:,:,:3], axis=2)[0,0]

    if use_cuda:
        ims = ims.cuda()
        t_ims = t_ims.cuda()
        viewpoints = viewpoints.cuda()
        gt_points = gt_points.cuda()
        gt_normals = gt_normals.cuda()

    # Forward
    graph.reset()
    sigma = next(sigma_scheme)
    pools = []
    for i in range(nIms):
        pools.append(FeaturePooling(t_ims[i]))

    for i in range(nQuery):
        query_viewpoint = get_random_viewpoint(distance).float().reshape((1, 7))
        if use_cuda:
            query_viewpoint = query_viewpoint.cuda()
        x_mu = model_gqn.sample(ims, viewpoints, query_viewpoint, sigma)
        # save_image(x_mu, args.output+"img_"+str(n)+"_"+str(i)+".png")
        x_mu = vgg_normalize(x_mu, device)
        pools.append(FeaturePooling(x_mu))
    pred_points = model_gcn(graph, pools)

    # Compute eval metrics
    _, loss_norm = chamfer_loss(pred_points[-1], gt_points.squeeze(), normalized=True)
    _, loss_unorm = chamfer_loss(pred_points[-1], gt_points.squeeze(), normalized=False)
    tot_loss_norm += loss_norm.item()
    tot_loss_unorm += loss_unorm.item()
    tot_f1_1 += f1_score(pred_points[-1], gt_points.squeeze(), threshold=tau)
    tot_f1_2 += f1_score(pred_points[-1], gt_points.squeeze(), threshold=2*tau)

    # Logs
    if n%log_step == 0:
        print("Batch", n)
        print("Normalized Chamfer loss so far", tot_loss_norm/(n+1))
        print("Unormalized Chamfer loss so far", tot_loss_unorm/(n+1))
        print("F1 score (tau=1e-4)", tot_f1_1/(n+1))
        print("F1 score (tau=2e-4)", tot_f1_2/(n+1))

    # Generate meshes
    if args.output is not None:
        graph.vertices = pred_points[5]
        graph.faces = graph.info[3][2]
        graph.to_obj(args.output + "plane_pred_block3_3_"
                        + str(n) + "_" + str(loss_norm.item()) + ".obj")
        graph.vertices = pred_points[3]
        graph.faces = graph.info[2][2]
        graph.to_obj(args.output + "plane_pred_block3_2_"
                        + str(n) + "_" + str(loss_norm.item()) + ".obj")
        graph.vertices = pred_points[1]
        graph.faces = graph.info[1][2]
        graph.to_obj(args.output + "plane_pred_block3_1_"
                        + str(n) + "_" + str(loss_norm.item()) + ".obj")
        graph.vertices = gt_points[0, :, :]
        graph.faces = []
        graph.to_obj(args.output + "plane_gt"
                        + str(n) + "_" + str(loss_norm.item()) + ".obj")
        print("Mesh plane_pred" + str(n) + "_" + str(loss_norm.item()) + " generated")

print("Final Normalized Chamfer loss:", tot_loss_norm/(n+1))
print("Final Unormalized Chamfer loss:", tot_loss_norm/(n+1))
print("Final F1 score (tau=1e-4) :", tot_f1_1/(n+1))
print("Final F1 score (tau=2e-4) :", tot_f1_2/(n+1))
