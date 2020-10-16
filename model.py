import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch_geometric.nn import GCNConv

from pool import FeaturePooling

class TowerRepresentation(nn.Module):
    def __init__(self, n_channels, v_dim, r_dim=256, pool=True):
        """
        Network that generates a condensed representation
        vector from a joint input of image and viewpoint.

        Employs the tower/pool architecture described in the paper.

        :param n_channels: number of color channels in input image
        :param v_dim: dimensions of the viewpoint vector
        :param r_dim: dimensions of representation
        :param pool: whether to pool representation
        """
        super(TowerRepresentation, self).__init__()
        # Final representation size
        self.r_dim = k = r_dim
        self.pool = pool

        self.conv1 = nn.Conv2d(n_channels, k, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(k, k, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(k, k//2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(k//2, k, kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(k + v_dim, k, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(k + v_dim, k//2, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(k//2, k, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(k, k, kernel_size=1, stride=1)

        self.avgpool  = nn.AvgPool2d(k//16)

    def forward(self, x, v):
        """
        Send an (image, viewpoint) pair into the
        network to generate a representation
        :param x: image
        :param v: viewpoint (x, y, z, cos(yaw), sin(yaw), cos(pitch), sin(pitch))
        :return: representation
        """
        # Increase dimensions
        v = v.view(v.size(0), -1, 1, 1)
        v = v.repeat(1, 1, self.r_dim // 16, self.r_dim // 16)

        # First skip-connected conv block
        skip_in  = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))

        x = F.relu(self.conv3(skip_in))
        x = F.relu(self.conv4(x)) + skip_out

        # Second skip-connected conv block (merged)
        skip_in = torch.cat([x, v], dim=1)
        skip_out  = F.relu(self.conv5(skip_in))

        x = F.relu(self.conv6(skip_in))
        x = F.relu(self.conv7(x)) + skip_out

        r = F.relu(self.conv8(x))

        if self.pool:
            r = self.avgpool(r)

        return r

class Conv2dLSTMCell(nn.Module):
    """
    2d convolutional long short-term memory (LSTM) cell.
    Functionally equivalent to nn.LSTMCell with the
    difference being that nn.Linear layers are replaced
    by nn.Conv2D layers.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: size of image kernel
    :param stride: length of kernel stride
    :param padding: number of pixels to pad with
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)

        self.forget = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.input  = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.output = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.state  = nn.Conv2d(in_channels, out_channels, **kwargs)

        self.transform = nn.Conv2d(out_channels, in_channels, **kwargs)

    def forward(self, input, states):
        """
        Send input through the cell.

        :param input: input to send through
        :param states: (hidden, cell) pair of internal state
        :return new (hidden, cell) pair
        """
        (hidden, cell) = states

        input = input + self.transform(hidden)

        forget_gate = torch.sigmoid(self.forget(input))
        input_gate  = torch.sigmoid(self.input(input))
        output_gate = torch.sigmoid(self.output(input))
        state_gate  = torch.tanh(self.state(input))

        # Update internal cell state
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)

        return hidden, cell

SCALE = 4 # Scale of image generation process

class GeneratorNetwork(nn.Module):
    """
    Network similar to a convolutional variational
    autoencoder that refines the generated image
    over a number of iterations.

    :param x_dim: number of channels in input
    :param v_dim: dimensions of viewpoint
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param L: number of density refinements
    :param share: whether to share cores across refinements
    """
    def __init__(self, x_dim, v_dim, r_dim, z_dim=64, h_dim=128, L=12, share=True):
        super(GeneratorNetwork, self).__init__()
        self.L = L
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.share = share

        # Core computational units
        kwargs = dict(kernel_size=5, stride=1, padding=2)
        inference_args = dict(in_channels=v_dim + r_dim + x_dim + h_dim, out_channels=h_dim, **kwargs)
        generator_args = dict(in_channels=v_dim + r_dim + z_dim, out_channels=h_dim, **kwargs)
        if self.share:
            self.inference_core = Conv2dLSTMCell(**inference_args)
            self.generator_core = Conv2dLSTMCell(**generator_args)
        else:
            self.inference_core = nn.ModuleList([Conv2dLSTMCell(**inference_args) for _ in range(L)])
            self.generator_core = nn.ModuleList([Conv2dLSTMCell(**generator_args) for _ in range(L)])

        # Inference, prior
        self.posterior_density = nn.Conv2d(h_dim, 2*z_dim, **kwargs)
        self.prior_density     = nn.Conv2d(h_dim, 2*z_dim, **kwargs)

        # Generative density
        self.observation_density = nn.Conv2d(h_dim, x_dim, kernel_size=1, stride=1, padding=0)

        # Up/down-sampling primitives
        self.upsample   = nn.ConvTranspose2d(h_dim, h_dim, kernel_size=SCALE, stride=SCALE, padding=0, bias=False)
        self.downsample = nn.Conv2d(x_dim, x_dim, kernel_size=SCALE, stride=SCALE, padding=0, bias=False)

    def forward(self, x, v, r):
        """
        Attempt to reconstruct x with corresponding
        viewpoint v and context representation r.

        :param x: image to send through
        :param v: viewpoint of image
        :param r: representation for image
        :return reconstruction of x and kl-divergence
        """
        batch_size, _, h, w = x.shape
        kl = 0

        # Downsample x, upsample v and r
        x = self.downsample(x)
        v = v.view(batch_size, -1, 1, 1).repeat(1, 1, h // SCALE, w // SCALE)
        if r.size(2) != h // SCALE:
            r = r.repeat(1, 1, h // SCALE, w // SCALE)

        # Reset hidden and cell state
        hidden_i = x.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))
        cell_i   = x.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))

        hidden_g = x.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))
        cell_g   = x.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))

        # Canvas for updating
        u = x.new_zeros((batch_size, self.h_dim, h, w))

        for l in range(self.L):
            # Prior factor (eta π network)
            p_mu, p_std = torch.chunk(self.prior_density(hidden_g), 2, dim=1)
            prior_distribution = Normal(p_mu, F.softplus(p_std))

            # Inference state update
            inference = self.inference_core if self.share else self.inference_core[l]
            hidden_i, cell_i = inference(torch.cat([hidden_g, x, v, r], dim=1), [hidden_i, cell_i])

            # Posterior factor (eta e network)
            q_mu, q_std = torch.chunk(self.posterior_density(hidden_i), 2, dim=1)
            posterior_distribution = Normal(q_mu, F.softplus(q_std))

            # Posterior sample
            z = posterior_distribution.rsample()

            # Generator state update
            generator = self.generator_core if self.share else self.generator_core[l]
            hidden_g, cell_g = generator(torch.cat([z, v, r], dim=1), [hidden_g, cell_g])

            # Calculate u
            u = self.upsample(hidden_g) + u

            # Calculate KL-divergence
            kl += kl_divergence(posterior_distribution, prior_distribution)

        x_mu = self.observation_density(u)

        return torch.sigmoid(x_mu), kl

    def sample(self, x_shape, v, r):
        """
        Sample from the prior distribution to generate
        a new image given a viewpoint and representation

        :param x_shape: (height, width) of image
        :param v: viewpoint
        :param r: representation (context)
        """
        h, w = x_shape
        batch_size = v.size(0)

        # Increase dimensions
        v = v.view(batch_size, -1, 1, 1).repeat(1, 1, h // SCALE, w // SCALE)
        if r.size(2) != h // SCALE:
            r = r.repeat(1, 1, h // SCALE, w // SCALE)

        # Reset hidden and cell state for generator
        hidden_g = v.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))
        cell_g = v.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))

        u = v.new_zeros((batch_size, self.h_dim, h, w))

        for _ in range(self.L):
            p_mu, p_log_std = torch.chunk(self.prior_density(hidden_g), 2, dim=1)
            prior_distribution = Normal(p_mu, F.softplus(p_log_std))

            # Prior sample
            z = prior_distribution.sample()

            # Calculate u
            hidden_g, cell_g = self.generator_core(torch.cat([z, v, r], dim=1), [hidden_g, cell_g])
            u = self.upsample(hidden_g) + u

        x_mu = self.observation_density(u)

        return torch.sigmoid(x_mu)

class GenerativeQueryNetwork(nn.Module):
    """
    Generative Query Network (GQN) as described
    in "Neural scene representation and rendering"
    [Eslami 2018].

    :param x_dim: number of channels in input
    :param v_dim: dimensions of viewpoint
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param L: Number of refinements of density
    """
    def __init__(self, x_dim, v_dim, r_dim, h_dim, z_dim, L=12):
        super(GenerativeQueryNetwork, self).__init__()
        self.r_dim = r_dim

        self.generator = GeneratorNetwork(x_dim, v_dim, r_dim, z_dim, h_dim, L)
        self.representation = TowerRepresentation(x_dim, v_dim, r_dim, pool=True)

    def forward(self, context_x, context_v, query_x, query_v):
        """
        Forward through the GQN.

        :param x: batch of context images [b, m, c, h, w]
        :param v: batch of context viewpoints for image [b, m, k]
        :param x_q: batch of query images [b, c, h, w]
        :param v_q: batch of query viewpoints [b, k]
        """
        # Merge batch and view dimensions.
        b, m, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape

        x = context_x.view((-1, *x_dims))
        v = context_v.view((-1, *v_dims))

        # representation generated from input images
        # and corresponding viewpoints
        phi = self.representation(x, v)

        # Seperate batch and view dimensions
        _, *phi_dims = phi.shape
        phi = phi.view((b, m, *phi_dims))

        # sum over view representations
        r = torch.sum(phi, dim=1)

        # Use random (image, viewpoint) pair in batch as query
        x_mu, kl = self.generator(query_x, query_v, r)

        # Return reconstruction and query viewpoint
        # for computing error
        return (x_mu, r, kl)

    def sample(self, context_x, context_v, query_v, sigma):
        """
        Sample from the network given some context and viewpoint.

        :param context_x: set of context images to generate representation
        :param context_v: viewpoints of `context_x`
        :param viewpoint: viewpoint to generate image from
        :param sigma: pixel variance
        """
        batch_size, n_views, _, h, w = context_x.shape

        _, _, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape

        x = context_x.view((-1, *x_dims))
        v = context_v.view((-1, *v_dims))

        phi = self.representation(x, v)

        _, *phi_dims = phi.shape
        phi = phi.view((batch_size, n_views, *phi_dims))

        r = torch.sum(phi, dim=1)

        x_mu = self.generator.sample((h, w), query_v, r)
        return x_mu

class VGG16(nn.Module):
    '''
    Pretrained VGG for image feature extraction
    '''
    def __init__(self):
        super(VGG16, self).__init__()
        model_conv = torchvision.models.vgg16(pretrained=True).features

        # Extract VGG feature maps conv_3, conv_4, conv_5
        layers = list(model_conv.children())
        self.conv3 = nn.Sequential(*layers[:-14])
        self.conv4 = nn.Sequential(*layers[:-7])
        self.conv5 = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv3(x), self.conv4(x), self.conv5(x)

class ResNet18(nn.Module):
    '''
    Pretrained ResNet for image feature extraction
    '''
    def __init__(self):
        super(ResNet18, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)

        # Extract ResNet feature maps layer2, layer3, layer4
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x_layer2 = self.layer2(x)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)
        return x_layer2, x_layer3, x_layer4

class DeformationBlock(nn.Module):
    '''
    Implement a mesh deformation block
    '''
    def __init__(self, feat_shape_dim):
        super(DeformationBlock, self).__init__()
        # self.conv1 = GCNConv(1280 + feat_shape_dim, 1024)
        self.conv1 = GCNConv(896 + feat_shape_dim, 1024)
        self.conv21 = GCNConv(1024, 512)
        self.conv22 = GCNConv(512, 256)
        self.conv23 = GCNConv(256, 128)
        self.conv2 = [self.conv21, self.conv22, self.conv23]
        self.conv3 = GCNConv(128, 3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge_index):
        '''
        Return 3D shape features (return[0]) and predicted 3D coordinates
        of the vertices (return[1])
        '''
        out = self.conv1(x, edge_index)
        out = self.relu(out)
        for i in range(len(self.conv2)):
            conv = self.conv2[i]
            out = conv(out, edge_index)
            out = self.relu(out)
        return out, self.conv3(out, edge_index)

class DeformationGNet(torch.nn.Module):
    '''
    Implement the full cascaded mesh deformation network
    '''
    def __init__(self):
        super(DeformationGNet, self).__init__()
        # self.feat_extr = VGG16()
        self.feat_extr = ResNet18()
        self.layer1 = DeformationBlock(3) # No shape features for block 1
        self.layer2 = DeformationBlock(128)
        self.layer3 = DeformationBlock(128)

    def forward(self, graph, pools):
        # Initial ellipsoid mesh
        elli_points = graph.vertices.clone()

        # Layer 1
        features = pools[0](elli_points, self.feat_extr)
        for i in range(1, 5):
            features = features + pools[i](elli_points, self.feat_extr)
        input = torch.cat((features, elli_points), dim=1)
        x, coord_1 = self.layer1(input, graph.adjacency_mat[0])
        graph.vertices = coord_1

        # Unpool graph
        x = graph.unpool(x)
        coord_1_1 = graph.vertices.clone()

        # Layer 2
        features = pools[0](graph.vertices, self.feat_extr)
        for i in range(1, 5):
            features = features + pools[i](graph.vertices, self.feat_extr)
        input = torch.cat((features, x), dim=1)
        x, coord_2 = self.layer2(input, graph.adjacency_mat[1])
        graph.vertices = coord_2

        # Unpool graph
        x = graph.unpool(x)
        coord_2_1 = graph.vertices.clone()

        # Layer 3
        features = pools[0](graph.vertices, self.feat_extr)
        for i in range(1, 5):
            features = features + pools[i](graph.vertices, self.feat_extr)
        input = torch.cat((features, x), dim=1)
        x, coord_3 = self.layer3(input, graph.adjacency_mat[2])
        graph.vertices = coord_3

        return elli_points, coord_1, coord_1_1, coord_2, coord_2_1, coord_3

    def get_nb_trainable_params(self):
        '''
        Return the number of trainable parameters
        '''
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

class ReconstructionNet(torch.nn.Module):
    '''
    Implement the full end-to-end small body reconstruction network
    '''
    def __init__(self, x_dim, v_dim, r_dim, h_dim, z_dim, L=12):
        super(ReconstructionNet, self).__init__()
        self.generator = GenerativeQueryNetwork(x_dim, v_dim, r_dim, h_dim, z_dim, L)
        self.deformation = DeformationGNet()

    def forward(self, context_x, context_v, query_x, query_v, graph):
        _, m, _ = context_x.shape
        x_mu, _, kl = self.generator(context_x, context_v, query_x, query_v)
        pools = []
        for i in range(m):
            pools.append(FeaturePooling(context_x[i])) # TODO Fix dimension
        pools.append(FeaturePooling(x_mu))

        pred_points = self.deformation(graph, pools)

        return pred_points, kl
