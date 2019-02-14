import matplotlib
import sys

matplotlib.use("Agg")
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from bavae import BAVAE
import h5py
import scipy as sp
import os
import pdb
import logging
import pylab as pl
from utils import export_scripts
from data_parser import read_face_data, FaceDataset
from optparse import OptionParser
import logging
import pickle
import time
import numpy as np

parser = OptionParser()
parser.add_option(
    "--data",
    dest="data",
    type=str,
    default="./data/faceplace/data_faces.h5",
    help="dataset path",
)
parser.add_option(
    "--outdir", dest="outdir", type=str, default="./out/avae7", help="output dir"
)
parser.add_option("--seed", dest="seed", type=int, default=0, help="seed")
parser.add_option(
    "--filts", dest="filts", type=int, default=32, help="number of convol filters"
)
parser.add_option("--zdim", dest="zdim", type=int, default=256, help="zdim")
parser.add_option(
    "--vy", dest="vy", type=float, default=2e-3, help="conditional norm lik variance"
)
parser.add_option("--lr", dest="lr", type=float, default=2e-4, help="learning rate")
parser.add_option("--bs", dest="bs", type=int, default=64, help="batch size")
parser.add_option(
    "--epoch_cb",
    dest="epoch_cb",
    type=int,
    default=10,
    help="number of epoch by which a callback (plot + dump weights) is executed",
)
parser.add_option(
    "--epochs", dest="epochs", type=int, default=51, help="total number of epochs"
)
parser.add_option("--debug", action="store_true", dest="debug", default=False)
(opt, args) = parser.parse_args()
opt_dict = vars(opt)

if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# output dir
wdir = os.path.join(opt.outdir, "weights")
fdir = os.path.join(opt.outdir, "plots")
if not os.path.exists(wdir):
    os.makedirs(wdir)
if not os.path.exists(fdir):
    os.makedirs(fdir)

# copy code to output folder
export_scripts(os.path.join(opt.outdir, "scripts"))

# create logfile
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(opt.outdir, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("opt = %s", opt)

# extract VAE settings and export
vae_cfg = {"nf": opt.filts, "zdim": opt.zdim, "vy": opt.vy}
pickle.dump(vae_cfg, open(os.path.join(opt.outdir, "vae.cfg.p"), "wb"))


def main():
    torch.manual_seed(opt.seed)

    if opt.debug:
        pdb.set_trace()

    # define VAE and optimizer
    vae = BAVAE(**vae_cfg).to(device)

    # optimizer
    optimizer_Enc = optim.Adam(vae.encode.parameters(), lr=0.0003)
    optimizer_Dec = optim.Adam(vae.decode.parameters(), lr=0.0003)
    optimizer_GDis = optim.Adam(vae.discrim.parameters(), lr=0.00003)
    optimizer_IDis = optim.Adam(vae.discrim.parameters(), lr=0.00003)

    # load data
    img, obj, view = read_face_data(opt.data)  # image, object, and view
    train_data = FaceDataset(img["train"], obj["train"], view["train"])
    val_data = FaceDataset(img["val"], obj["val"], view["val"])
    train_queue = DataLoader(train_data, batch_size=opt.bs, shuffle=True)
    val_queue = DataLoader(val_data, batch_size=opt.bs, shuffle=False)

    for epoch in range(opt.epochs):
        train_ep(vae, train_queue, optimizer_Enc, optimizer_Dec, optimizer_GDis, optimizer_IDis, gamma=1)
        eval_ep(vae, val_queue, gamma=1)


def train_ep(vae, train_queue, optimizer_Enc, optimizer_Dec, optimizer_GDis, optimizer_IDis, gamma=1):
    vae.train()

    for batch_i, data in enumerate(train_queue):
        batch_size = len(data[0])

        # forward
        y = data[0]
        eps = Variable(torch.randn(y.shape[0], 256), requires_grad=False)
        y, eps = y.to(device), eps.to(device)
        idloss, gdloss = vae.forward(y, eps)

        enc_loss = - idloss
        dec_loss = - gdloss

        optimizer_IDis.zero_grad()
        idloss.backward(retain_graph=True)
        optimizer_IDis.step()

        optimizer_GDis.zero_grad()
        gdloss.backward(retain_graph=True)
        optimizer_GDis.step()

        optimizer_Dec.zero_grad()
        dec_loss.backward(retain_graph=True)
        optimizer_Dec.step()

        optimizer_Enc.zero_grad()
        enc_loss.backward()
        optimizer_Enc.step()


def eval_ep(vae, val_queue, gamma):
    vae.eval()

    with torch.no_grad():
        for batch_i, data in enumerate(val_queue):
            batch_size = len(data[0])

            # forward
            y = data[0]
            eps = Variable(torch.randn(y.shape[0], 256), requires_grad=False)
            y, eps = y.to(device), eps.to(device)
            idloss, gdloss = vae.forward(y, eps)

            enc_loss = - idloss
            dec_loss = - gdloss


if __name__ == "__main__":
    main()
