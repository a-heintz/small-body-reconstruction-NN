import os
import argparse
import torch
import pickle
import torch.utils.data as utils
import torch.optim as optim
from torch.distributions import Normal
import time
import numpy as np

from graph import Graph
from model import ReconstructionNet
from pool import FeaturePooling
from metrics import loss_function
from data import CustomDatasetFolder

# Args
parser = argparse.ArgumentParser(description='Pixel2Mesh training script')
parser.add_argument('--data', type=str, default=None, metavar='D',
                    help="folder where data is located.")
parser.add_argument('--epochs', type=int, default=100, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=3e-5, metavar='LR',
                    help='learning rate (default: 3e-5)')
parser.add_argument('--log_step', type=int, default=100, metavar='LS',
                    help='how many batches to wait before logging training status (default: 100)')
parser.add_argument('--saving_step', type=int, default=1000, metavar='S',
                    help='how many batches to wait before saving model (default: 1000)')
parser.add_argument('--experiment', type=str, default='./model/', metavar='E',
                    help='folder where model and optimizer are saved.')
parser.add_argument('--load_model', type=str, default=None, metavar='M',
                    help='model file to load to continue training.')
parser.add_argument('--load_optimizer', type=str, default=None, metavar='O',
                    help='model file to load to continue training.')
args = parser.parse_args()

# Cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model
if args.load_model is not None: # Continue training
    state_dict = torch.load(args.load_model, map_location=device)
    model_gcn = ReconstructionNet(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8)
    model_gcn.load_state_dict(state_dict)
else:
    model_gcn = ReconstructionNet(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8)

# Optimizer
if args.load_optimizer is not None:
    state_dict_opt = torch.load(args.load_optimizer, map_location=device)
    optimizer = optim.Adam(model_gcn.parameters(), lr=args.lr)
    optimizer.load_state_dict(state_dict_opt)
else:
    optimizer = optim.Adam(model_gcn.parameters(), lr=args.lr)
model_gcn.train()

# Graph
graph = Graph("./ellipsoid/init_info.pickle")

# Data Loader
folder = CustomDatasetFolder(args.data, extensions = ["dat"])
train_loader = torch.utils.data.DataLoader(folder, batch_size=1, shuffle=True)

# Param
nb_epochs = args.epochs
log_step = args.log_step
saving_step = args.saving_step
curr_loss = 0

# To GPU
if use_cuda:
    print('Using GPU', flush=True)
    model_gcn.cuda()
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
else:
    print('Using CPU', flush=True)

print("nb trainable param", model_gcn.get_nb_trainable_params(), flush=True)

# Train
for epoch in range(1, nb_epochs+1):
    for n, data in enumerate(train_loader):
        ims, gt_points, gt_normals = data
        ims = np.transpose(ims, (1, 0, 2, 3, 4))

        if use_cuda:
            ims = ims.cuda()
            gt_points = gt_points.cuda()
            gt_normals = gt_normals.cuda()

        m, b, *x_dims = ims.shape
        indices = np.arange(m)
        context_idx, query_idx = indices[:-1], indices[-1]
        context_ims = ims[context_idx]
        query_ims = ims[query_idx]

        # Forward
        graph.reset()
        optimizer.zero_grad()
        pred_points, x_mu, kl = model_gcn(context_ims, context_views, query_ims, query_views, graph) #TODO Viewpoints

        # Loss
        loss = loss_function(pred_points, gt_points.squeeze(),
                                          gt_normals.squeeze(), graph)

        ll = Normal(x_mu, sigma).log_prob(query_ims)

        likelihood     = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
        kl_divergence  = torch.mean(torch.sum(kl, dim=[1, 2, 3]))
        elbo = likelihood - kl_divergence
        loss = loss - elbo * 1e-4

        # Backward
        loss.backward()
        optimizer.step()

        curr_loss += loss

        # Log
        if (n+1)%log_step == 0:
            print("Epoch", epoch, flush=True)
            print("Batch", n+1, flush=True)
            print(" Loss:", curr_loss.data.item()/log_step, flush=True)
            curr_loss = 0

        # Save
        if (n+1)%saving_step == 0:
            model_file = args.experiment + "model_" + str(n+1) + ".pth"
            optimizer_file = args.experiment + "optimizer_" + str(n+1) + ".pth"
            torch.save(model_gcn.state_dict(), model_file)
            torch.save(optimizer.state_dict(), optimizer_file)
            print("Saved model to " + model_file, flush=True)
