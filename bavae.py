import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import h5py
import scipy as sp
import os
import pdb
import pylab as pl

def f_act(x, act="elu"):
    if act == "relu":
        return F.relu(x)
    elif act == "elu":
        return F.elu(x)
    elif act == "linear":
        return x
    else:
        return None


class GenerativeDiscriminator(nn.Module):
    def __init__(self, img_size=128, nf=32, zdim=256, steps=5, colors=3, act="elu", vy=1e-3, gamma = 0, ks=3):
        super(GenerativeDiscriminator, self).__init__()
        # conv cells discriminator
        self.disconv = nn.ModuleList()
        cell = Conv2dCellDown(colors, nf, ks, act)
        self.disconv += [cell]
        for i in range(steps - 1):
            cell = Conv2dCellDown(nf, nf, ks, act)
            self.disconv += [cell]
        cell = nn.Conv2d(nf, 1, 4)
        self.disconv += [cell]

    def forward(self, x):
        for cell in self.disconv:
            x = cell(x)
        return x


class GenerativeDiscriminatorLoss(torch.nn.Module):

    def __init__(self):
        super(GenerativeDiscriminatorLoss, self).__init__()

    def forward(self, real, fake):
        loss_fake = 1 - fake
        loss_real = -torch.exp(-real)
        return -(loss_fake+loss_real)


class InferenceDiscriminator(nn.Module):
    def __init__(self, img_size=128, nf=32, zdim=256, steps=5, colors=3, act="elu", vy=1e-3, gamma = 0, ks=3):
        super(InferenceDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(zdim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


class InferenceDiscriminatorLoss(torch.nn.Module):

    def __init__(self):
        super(InferenceDiscriminatorLoss, self).__init__()

    def forward(self, real, fake):
        loss_fake = -torch.exp(fake-1)
        loss_real = real
        return -(loss_fake+loss_real)


class Conv2dCellDown(nn.Module):
    def __init__(self, ni, no, ks=3, act="elu"):
        super(Conv2dCellDown, self).__init__()
        self.act = act
        self.conv1 = nn.Conv2d(ni, no, kernel_size=ks, stride=1, padding=1)
        self.conv2 = nn.Conv2d(no, no, kernel_size=ks, stride=2, padding=1)

    def forward(self, x):
        x = f_act(self.conv1(x), act=self.act)
        x = f_act(self.conv2(x), act=self.act)
        return x


class Conv2dCellUp(nn.Module):
    def __init__(self, ni, no, ks=3, act1="elu", act2="elu"):
        super(Conv2dCellUp, self).__init__()
        self.act1 = act1
        self.act2 = act2
        self.conv1 = nn.Conv2d(ni, no, kernel_size=ks, stride=1, padding=1)
        self.conv2 = nn.Conv2d(no, no, kernel_size=ks, stride=1, padding=1)

    def forward(self, x):
        x = F.upsample(x, scale_factor=2)
        x = f_act(self.conv1(x), act=self.act1)
        x = f_act(self.conv2(x), act=self.act2)
        return x


class Encoder(nn.Module):
    def __init__(self, img_size=128, nf=32, zdim=256, steps=5, colors=3, act="elu", vy=1e-3, gamma = 0, ks=3):
        super(Encoder, self).__init__()
        self.red_img_size = img_size // (2 ** steps)
        self.nf = nf
        self.gamma = gamma
        self.size_flat = self.red_img_size ** 2 * nf
        self.K = img_size ** 2 * colors
        self.dense_z = nn.Linear(self.size_flat, zdim)

        # conv cells encoder
        self.econv = nn.ModuleList()
        cell = Conv2dCellDown(colors, nf, ks, act)
        self.econv += [cell]
        for i in range(steps - 1):
            cell = Conv2dCellDown(nf, nf, ks, act)
            self.econv += [cell]

    def forward(self, x):
        for ic, cell in enumerate(self.econv):
            x = cell(x)
        x = x.view(-1, self.size_flat)
        z = self.dense_z(x)
        return z


class Decoder(nn.Module):
    def __init__(self, img_size=128, nf=32, zdim=256, steps=5, colors=3, act="elu", vy=1e-3, gamma = 0, ks=3):
        super(Decoder, self).__init__()
        self.red_img_size = img_size // (2 ** steps)
        self.nf = nf
        self.gamma = gamma
        self.size_flat = self.red_img_size ** 2 * nf
        self.K = img_size ** 2 * colors
        self.dense_dec = nn.Linear(zdim, self.size_flat)
        # conv cells decoder
        self.dconv = nn.ModuleList()
        for i in range(steps - 1):
            cell = Conv2dCellUp(nf, nf, ks, act1=act, act2=act)
            self.dconv += [cell]
        cell = Conv2dCellUp(nf, colors, ks, act1=act, act2="linear")
        self.dconv += [cell]

    def forward(self, x):
        x = self.dense_dec(x)
        x = x.view(-1, self.nf, self.red_img_size, self.red_img_size)
        for cell in self.dconv:
            x = cell(x)
        return x


class BAVAE(nn.Module):
    def __init__(
        self, img_size=128, nf=32, zdim=256, steps=5, colors=3, act="elu", vy=1e-3, gamma = 0
    ):

        super(BAVAE, self).__init__()

        # store useful stuff
        self.red_img_size = img_size // (2 ** steps)
        self.nf = nf
        self.gamma = gamma
        self.size_flat = self.red_img_size ** 2 * nf
        self.K = img_size ** 2 * colors
        self.zdim = zdim
        ks = 3

        self.encode = Encoder()
        self.decode = Decoder()
        self.gdiscrim =  GenerativeDiscriminator()
        self.idiscrim = InferenceDiscriminator()
        self.gdiscrimloss = GenerativeDiscriminatorLoss()
        self.idiscrimloss = InferenceDiscriminatorLoss()

    def forward(self, x, eps):

        #inference
        inf_d_real = self.idiscrim(eps)
        infered_z = self.encode(x)
        inf_d_fake = self.idiscrim(infered_z)
        idloss = self.idiscrimloss(inf_d_real, inf_d_fake)

        #generative process
        gen_d_real = self.gdiscrim(x)
        generated_x = self.decode(infered_z)
        gen_d_fake = self.gdiscrim(generated_x)
        gdloss = self.gdiscrimloss(gen_d_real, gen_d_fake)

        return idloss, gdloss

