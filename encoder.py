import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class Encoder(torch.nn.Module):

    def __init__(self, layer_type, layer_params, activation_type, maxpool,
                 train_bn_scaling, noise_level, use_cuda):

        super(Encoder, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.layer_type = layer_type
        self.layer_params = layer_params
        self.activation_type = activation_type
        self.maxpool = maxpool
        self.train_bn_scaling = train_bn_scaling
        self.noise_level = noise_level
        self.use_cuda = use_cuda

        if self.layer_type == 'Conv2D':
            self.layer = torch.nn.Conv2D(d_in, d_out, layer_params[0], layer_params[1], bias=False)
            self.layer.weight.data = torch.randn(self.layer.weight.data.size()) / np.sqrt(d_in * layer_params[0] ** 2)
            self.bn_normalize_clean = torch.nn.BatchNorm2d(d_out, affine=False)
            self.bn_normalize = torch.nn.BatchNorm2d(d_out, affine=False)
        else:
            self.layer = torch.nn.Linear(d_in, d_out, bias=False)
            self.layer.weight.data = torch.randn(self.layer.weight.data.size()) / np.sqrt(d_in)
            self.bn_normalize_clean = torch.nn.BatchNorm1d(d_out, affine=False)
            self.bn_normalize = torch.nn.BatchNorm1d(d_out, affine=False)

        if activation_type == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation_type == 'softmax':
            self.activation = torch.nn.Softmax()
        elif activation_type == None:
            self.activation == torch.nn.Identity()
        else:
            raise ValueError("invalid Acitvation type")

        # Batch Normalization Params
        if self.use_cuda:
            self.bn_beta = Parameter(torch.cuda.FloatTensor(1, d_out, ))
        else:
            self.bn_beta = Parameter(torch.FloatTensor(1, d_out))
        self.bn_beta.data.zero_()

        if self.train_bn_scaling:
            # batch-normalization scaling
            if self.use_cuda:
                self.bn_gamma = Parameter(torch.cuda.FloatTensor(1, d_out))
                self.bn_gamma.data = torch.ones(self.bn_gamma.size()).cuda()
            else:
                self.bn_gamma = Parameter(torch.FloatTensor(1, d_out))
                self.bn_gamma.data = torch.ones(self.bn_gamma.size())

        # buffer for z_pre, z which will be used in decoder cost
        self.buffer_z_pre = None
        self.buffer_z = None
        # buffer for tilde_z which will be used by decoder for reconstruction
        self.buffer_tilde_z = None

    def bn_gamma_beta(self, x):  # returns x * gamma + B
        if self.use_cuda:
            ones = Parameter(torch.ones(x.size()[0], 1).cuda())
        else:
            ones = Parameter(torch.ones(x.size()[0], 1))
        t = x + ones.mm(self.bn_beta)
        if self.train_bn_scaling:
            t = torch.mul(t, ones.mm(self.bn_gamma))
        return t

    def forward_clean(self, h):
        z_pre = self.layer(h)
        # Store z_pre, z to be used in calculation of reconstruction cost
        self.buffer_z_pre = z_pre.detach().clone()
        z = self.bn_normalize_clean(z_pre)
        self.buffer_z = z.detach().clone()
        h = self.activation(z_gb)
        return h
