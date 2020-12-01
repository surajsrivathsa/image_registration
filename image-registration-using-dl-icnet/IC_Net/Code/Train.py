# !/usr/bin/python
# coding=utf-8
import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
from torch.autograd import Variable
from Models import ModelFlow_stride, SpatialTransform,antifoldloss,mse_loss,smoothloss
from Functions import Dataset, generate_grid, padding, train_padding
import torch.utils.data as Data
import SimpleITK as sitk
parser = ArgumentParser()
parser.add_argument("--lr", type=float, 
                    dest="lr", default=5e-4,help="learning rate") 
parser.add_argument("--iteration", type=int, 
                    dest="iteration", default=20001,
                    help="number of total iterations")
parser.add_argument("--inverse", type=float, 
                    dest="inverse", default=0.05,
                    help="Inverse consistent：suggested range 0.001 to 0.1")
parser.add_argument("--antifold", type=float, 
                    dest="antifold", default=100000,
                    help="Anti-fold loss: suggested range 100000 to 1000000")
parser.add_argument("--smooth", type=float, 
                    dest="smooth", default=0.5,
                    help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=4000, 
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8, 
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath", default='../Dataset', 
                    help="data path for training images")
opt = parser.parse_args()

lr = opt.lr
iteration = opt.iteration
start_channel = opt.start_channel
inverse = opt.inverse
antifold = opt.antifold
n_checkpoint = opt.checkpoint
smooth = opt.smooth
datapath = opt.datapath

def train():
    torch.cuda.empty_cache()
    device = torch.device("cuda:6")
    model =ModelFlow_stride(2,3,start_channel).cuda(device)
    # model =ModelFlow_stride(2,3,start_channel).cpu()
    loss_similarity =mse_loss
    loss_inverse = mse_loss
    loss_antifold = antifoldloss
    loss_smooth = smoothloss
    transform = SpatialTransform().cuda(device)
    # transform = SpatialTransform().cpu()
    for param in transform.parameters():
        param.requires_grad = False
        param.volatile=True
    names = glob.glob(datapath + '/*.gz')
    grid = generate_grid(imgshape)
    grid = Variable(torch.from_numpy(np.reshape(grid, (1,) + grid.shape))).cuda(device).float()
    # grid = Variable(torch.from_numpy(np.reshape(grid, (1,) + grid.shape))).cpu().float()


    print(grid.type())
    optimizer = torch.optim.Adam(model.parameters(),lr=lr) 
    model_dir = '../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((5,iteration))
    training_generator = Data.DataLoader(Dataset(names,iteration,True), batch_size=1,
                        shuffle=False, num_workers=0)
    step=0
    for  X,Y in training_generator:
        X = X.cuda(device).float()
        # X = X.cpu().float()
        Y = Y.cuda(device).float()
        # Y = Y.cpu().float()
        # # X = sitk.GetArrayFromImage(sitk.ReadImage(X, sitk.sitkFloat32))
        # # Y = sitk.GetArrayFromImage(sitk.ReadImage(Y, sitk.sitkFloat32))
        #X = Variable(X).cuda(device).float()
        #Y = Variable(Y).cuda(device).float()
        X,Y = train_padding(X,Y)  #added today
       
        X = X.cuda(device).float() #added as the values returned after padding was not cuda
        Y = Y.cuda(device).float()



        F_xy = model(X,Y)
        F_yx = model(Y,X)
        X_Y = transform(X,F_xy.permute(0,2,3,4,1)*range_flow,grid)
        Y_X = transform(Y,F_yx.permute(0,2,3,4,1)*range_flow,grid)
        F_xy_ = transform(-F_xy,F_xy.permute(0,2,3,4,1)*range_flow,grid)
        F_yx_ = transform(-F_yx,F_yx.permute(0,2,3,4,1)*range_flow,grid)
        loss1 = loss_similarity(Y,X_Y) + loss_similarity(X,Y_X)
        loss2 = loss_inverse(F_xy*range_flow,F_xy_*range_flow) + loss_inverse(F_yx*range_flow,F_yx_*range_flow)
        loss3 =  loss_antifold(F_xy*range_flow) + loss_antifold(F_yx*range_flow)
        loss4 =  loss_smooth(F_xy*range_flow) + loss_smooth(F_yx*range_flow)
        loss = loss1+inverse*loss2 + antifold*loss3 + smooth*loss4
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        #lossall[:,step] = np.array([loss.data[0],loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0]])
        lossall[:,step] = np.array([loss.data,loss1.data,loss2.data,loss3.data,loss4.data])
        #sys.stdout.write("\r" + 'step "{0}" -> training loss "{1:.4f}" - sim "{2:.4f}" - inv "{3:.4f}" - ant "{4:.4f}" -smo "{5:.4f}" '.format(step, loss.data[0],loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0]))
        sys.stdout.write("\r" + 'step "{0}" -> training loss "{1:.4f}" - sim "{2:.4f}" - inv "{3:.4f}" - ant "{4:.4f}" -smo "{5:.4f}" '.format(step, loss.data,loss1.data,loss2.data,loss3.data,loss4.data))
        sys.stdout.flush()
        if(step % n_checkpoint == 0):
            modelname = model_dir + '/' + str(step) + '.pth'
            torch.save(model.state_dict(), modelname)
        step+=1
        torch.cuda.empty_cache()
        del X
        del Y
    np.save(model_dir+'/loss.npy',lossall)


# imgshape = (144, 192, 160)
imgshape = (128, 128, 128)

range_flow = 7
train()
