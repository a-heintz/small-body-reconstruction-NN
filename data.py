import torch
import pickle
import numpy as np
import os
from skimage import io
from stl import mesh
import math
from scipy.spatial.transform import Rotation

class CustomDatasetFolder(torch.utils.data.Dataset):
    '''
    Data reader
    '''
    def __init__(self, root, extensions, dimension, print_ref=False):
        self.samples = self._make_dataset(root, extensions)
        self.root = root
        self.dimension = dimension
        if len(self.samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        # Normalization for VGG
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((-1, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((-1, 1, 1))
        # Print 3D model _ref
        self.print_ref = print_ref

    def __getitem__(self, index):
        path = self.samples[index]
        ims, viewpoints, points, normals = self._loader(path)

        # Apply small transform
        ims = ims.astype(float)/255.0
        ims = np.transpose(ims, (0, 3, 1, 2))
        transformed_ims = (ims - self.mean)/self.std

        return torch.from_numpy(ims).float(), \
               torch.from_numpy(transformed_ims).float(), \
               torch.from_numpy(viewpoints).float(), \
               torch.from_numpy(points).float(), \
               torch.from_numpy(normals).float()

    def __len__(self):
        return len(self.samples)

    def _loader(self, path):
        if self.print_ref:
            print(path)
        stl_indices = np.load(self.root + 'state_files/asteroid_choice.npy')
        stl_index = int(path[-8:-5])
        orbits_pos = np.load(self.root + 'state_files/orbits_positions.npy')
        orbits_att = np.load(self.root + 'state_files/orbits_attitudes.npy')
        img_indices = np.arange(40)
        np.random.shuffle(img_indices)
        ims = []
        viewpoints = []
        for i in range(self.dimension):
            ii = img_indices[i]
            img_path = path
            if ii < 10:
                img_path += "0"
            img_path += str(ii) + ".png"
            im = io.imread(img_path)
            im[np.where(im[:, :, 3] == 0)] = 255
            im = im[:, :, :3].astype(np.float32)
            ims.append(im)
            viewpoint = np.zeros(7)
            viewpoint[:3] = orbits_pos[stl_index, ii, :]
            viewpoint[3:] = orbits_att[stl_index, ii, :]
            viewpoints.append(viewpoint)
        stl_files = ['bennu.stl', 'itokawa.stl', 'mithra.stl', 'toutatis.stl']
        my_mesh = mesh.Mesh.from_file(self.root + 'stl_files/' + stl_files[stl_indices[stl_index]])
        normals = my_mesh.normals.astype(float)
        npy_files = ['bennu.npy', 'itokawa.npy', 'mithra.npy', 'toutatis.npy']
        points = np.load(self.root + 'stl_files/' + npy_files[stl_indices[stl_index]])
        return np.asarray(ims), np.asarray(viewpoints), np.asarray(points), np.asarray(normals)

    def _make_dataset(self, dir, extensions):
        paths = []
        stl_indices = np.load(dir + 'state_files/asteroid_choice.npy')
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                item_visited = ""
                for fname in sorted(fnames):
                    if self._has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = path[:-6] # strip away **.png
                        stl_index = int(item[-8:-5])
                        if item != item_visited and stl_indices[stl_index] != 2:
                        # if item != item_visited:
                            item_visited = item
                            paths.append(item)
        return paths

    def _has_file_allowed_extension(self, filename, extensions):
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in extensions)

def dcm2quat(dcm):
    return Rotation.from_dcm(dcm).as_quat()

def R_2vect(vector_orig, vector_fin):
    R = np.zeros((3,3))
    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / np.linalg.norm(vector_orig)
    vector_fin = vector_fin / np.linalg.norm(vector_fin)
    # The rotation axis (normalised).
    axis = np.cross(vector_orig, vector_fin)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len
    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]
    # The rotation angle.
    angle = math.acos(np.dot(vector_orig, vector_fin))
    # Trig functions (only need to do this maths once!).
    ca = math.cos(angle)
    sa = math.sin(angle)
    # Calculate the rotation matrix elements.
    R[0,0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    R[0,1] = -z*sa + (1.0 - ca)*x*y
    R[0,2] = y*sa + (1.0 - ca)*x*z
    R[1,0] = z*sa+(1.0 - ca)*x*y
    R[1,1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    R[1,2] = -x*sa+(1.0 - ca)*y*z
    R[2,0] = -y*sa+(1.0 - ca)*x*z
    R[2,1] = x*sa+(1.0 - ca)*y*z
    R[2,2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)
    return R

def rotationMatrixToEulerAngles(R) :
    #assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def get_random_xyz(distance) :
  v = np.random.randn(3)
  unit_v = v / np.linalg.norm(v)
  return unit_v * distance

def get_qs(posistion):
  vector_orig = [0,0,1]
  vec_orig_flip = np.array(vector_orig)
  R = R_2vect(vec_orig_flip, posistion)
  return dcm2quat(R)

def get_random_viewpoint(distance):
  pos = get_random_xyz(distance)
  att = get_qs(pos)
  arr = np.hstack((pos, att))
  return torch.from_numpy(arr)

def vgg_transform(img):
    img = img.detach().cpu()
    img = img.numpy().astype(float)/255.0
    # Normalization for VGG
    mean = np.array([0.485, 0.456, 0.406]).reshape((-1, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((-1, 1, 1))
    return torch.from_numpy((img - mean)/std)