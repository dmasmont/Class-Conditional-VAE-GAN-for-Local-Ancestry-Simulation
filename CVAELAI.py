from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np


# Useful URLs:
# https://github.com/pytorch/examples/blob/master/vae/main.py
# https://github.com/jojonki/AutoEncoders/blob/master/cvae.ipynb
# https://github.com/altosaar/variational-autoencoder/blob/master/train_variational_autoencoder_pytorch.py

class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size, hidden_size=400, use_batch_norm=False):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size
        self.hidden_size = hidden_size

        # encode
        self.fc1  = nn.Linear(feature_size + class_size, self.hidden_size)
        self.fc21 = nn.Linear(self.hidden_size, latent_size)
        self.fc22 = nn.Linear(self.hidden_size, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, feature_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # BatchNorms
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.bn_fc1 = nn.BatchNorm1d(self.hidden_size)
            self.bn_fc3 = nn.BatchNorm1d(self.hidden_size)


    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''

        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        _h1 = self.relu(self.fc1(inputs))
        if self.use_batch_norm:
            h1 = self.bn_fc1(_h1)
        else:
            h1 = _h1
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps*std + mu
        else:
            return mu

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        _h3 = self.relu(self.fc3(inputs))
        if self.use_batch_norm:
            h3 = self.bn_fc3(_h3)
        else:
            h3 = _h3
        out = self.fc4(h3)

        return out

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, c), mu, logvar


class Discriminator(nn.Module):
    def __init__(self, window_size, use_batch_norm=True, is_class_conditional=True, num_classes=3):
        super(Discriminator, self).__init__()

        self.is_class_conditional = is_class_conditional

        self.hidden_size = 200

        # encode
        input_size = window_size if not is_class_conditional else (window_size + num_classes)
        self.fc1  = nn.Linear(input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 1)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # BatchNorms
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.bn_fc1 = nn.BatchNorm1d(self.hidden_size)


    def forward(self, x, c=None):
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        if self.is_class_conditional:
            inputs = torch.cat([x, c], 1)  # (bs, feature_size+class_size)
        else:
            inputs = x

        _h1 = self.relu(self.fc1(inputs))
        if self.use_batch_norm:
            h1 = self.bn_fc1(_h1)
        else:
            h1 = _h1
        h2 = self.fc2(h1)
        out = self.sigmoid(h2)
        return out


class CVAELAI(nn.Module):
    def __init__(self, window_size, input_dimension, class_size=3, hidden_size=400, embedding_size=50, use_batch_norm=False, is_residual=True, residual_avg=None, residual_var=None, is_GAN=False):
        super(CVAELAI, self).__init__()

        self.windows_size = window_size
        self.input_dimension = input_dimension
        self.num_windows = int(np.floor(self.input_dimension / self.windows_size))

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_classes = class_size

        self.CVAEList = nn.ModuleList()

        if is_GAN:
            self.DiscriminatorList = nn.ModuleList()

        self.use_batch_norm = use_batch_norm
        self.is_residual = is_residual
        self.residual_constant = nn.Parameter(residual_avg, requires_grad=False) if residual_avg is not None else None
        self.residual_var = nn.Parameter(residual_var, requires_grad=False) if residual_var is not None else None

        for i in range(self.num_windows):
            if i == int(self.num_windows-1):
                _window_size = self.windows_size
                _window_size += np.remainder(self.input_dimension, self.windows_size)
            else:
                _window_size = self.windows_size

            cvae = CVAE(_window_size, self.embedding_size, self.num_classes, hidden_size=self.hidden_size, use_batch_norm=self.use_batch_norm)
            self.CVAEList.append(cvae)

            dis = Discriminator(_window_size)
            self.DiscriminatorList.append(dis)


    def forward(self, x, c):

        if self.is_residual:
            x = x - self.residual_constant
            if self.residual_var is not None:
                x = x/self.residual_var

        outputs_cvae = []
        outputs_mu = []
        outputs_logvar = []
        for j, cvae in enumerate(self.CVAEList):
            if j == len(self.CVAEList)-1:
                _x = x[:, j*self.windows_size:]
            else:
                _x = x[:, j * self.windows_size : (j + 1) * self.windows_size]
            _c = c[:,j,:]

            out_cvae, mu, logvar = cvae(_x, _c)
            outputs_cvae.append(out_cvae)
            outputs_mu.append(mu)
            outputs_logvar.append(logvar)

        output_mu = torch.stack(outputs_mu,dim=2)
        output_logvar = torch.stack(outputs_logvar,dim=2)
        output_decoder = torch.cat(outputs_cvae, dim=1)


        if self.is_residual:
            res_out = (x, output_decoder)
            output_decoder = output_decoder + self.residual_constant
            if self.residual_var is not None:
                output_decoder = output_decoder * self.residual_var

            return output_decoder, output_mu, output_logvar, res_out
        else:
            res_out = (x, output_decoder)
            return output_decoder, output_mu, output_logvar, res_out


    def simulate(self, z=None, c=None, var=1.0, batch_size=1, device=None, single_ancestry=True, use_ancestry=None):
        if z is None:
            z = torch.randn(batch_size, len(self.CVAEList), self.embedding_size).to(device)
            z *= var

        if c is None:
            if not single_ancestry:
                c = torch.randn(batch_size, len(self.CVAEList), self.num_classes).to(device)
            else:
                if use_ancestry is None:
                    anc = torch.randint(0,2,(batch_size, len(self.CVAEList)), device=device)
                else:
                    anc = torch.zeros((batch_size, len(self.CVAEList)), device=device, dtype=torch.long) + use_ancestry
                c = F.one_hot(anc, num_classes=3).float()

        outputs_cvae = []
        outputs_c = []
        for j, cvae in enumerate(self.CVAEList):
            _z = z[:, j, :]
            _c_rand = c[:, j, :]

            argmax_out = torch.argmax(_c_rand, dim=1)
            _c = F.one_hot(argmax_out, num_classes=3).float()

            out_cvae  = cvae.decode(_z, _c)
            outputs_cvae.append(out_cvae)
            outputs_c.append(_c)

        output_c = torch.stack(outputs_c,dim=2)
        output_decoder = torch.cat(outputs_cvae, dim=1)

        if self.is_residual:
            output_decoder = output_decoder + self.residual_constant
            if hasattr(self, 'residual_var'):
                if self.residual_var is not None:
                    output_decoder = output_decoder * self.residual_var

        return output_decoder, output_c


    def forward_discriminator(self, x, c):
        outputs_dis = []

        if self.is_residual:
            x = x - self.residual_constant
            if self.residual_var is not None:
                x = x/self.residual_var

        for j, dis in enumerate(self.DiscriminatorList):
            if j == len(self.DiscriminatorList)-1:
                _x = x[:, j*self.windows_size:]
            else:
                _x = x[:, j * self.windows_size : (j + 1) * self.windows_size]
            _c = c[:, :, j]
            out_dis = dis(_x, _c)
            outputs_dis.append(out_dis)

        output_dis = torch.stack(outputs_dis,dim=1).squeeze(dim=2)
        return output_dis
