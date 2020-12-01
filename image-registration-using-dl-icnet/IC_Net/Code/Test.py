import os
from argparse import ArgumentParser
import numpy as np
import torch
from torch.autograd import Variable
from Models import ModelFlow_stride,SpatialTransform
from Functions import generate_grid, load_5D, save_img, save_flow, padding
import timeit
import SimpleITK as sitk

parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='../model.pth',
                    help="frequency of saving models")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='../Result',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8, 
                    help="number of start channels")
parser.add_argument("--fixed", type=str,
                    dest="fixed", default='../Dataset/image_A.nii.gz', 
                    help="fixed image")
parser.add_argument("--moving", type=str,
                    dest="moving", default='../Dataset/image_B.nii.gz', 
                    help="moving image")


opt = parser.parse_args()
savepath = opt.savepath
if not os.path.isdir(savepath):
    os.mkdir(savepath)

def test():
    device = torch.device("cuda:6")
    model =ModelFlow_stride(2,3,opt.start_channel).cuda(device)
    # model =ModelFlow_stride(2,3,opt.start_channel).cpu()
    # transform = SpatialTransform().cpu()

    transform = SpatialTransform().cuda(device)
    # model.load_state_dict(torch.load(opt.modelpath))
    # model.load_state_dict(torch.load(opt.modelpath, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(opt.modelpath, map_location=device))

    model.eval()
    transform.eval()
    grid = generate_grid(imgshape)
    grid = Variable(torch.from_numpy(np.reshape(grid, (1,) + grid.shape))).cuda(device).float()
    # grid = Variable(torch.from_numpy(np.reshape(grid, (1,) + grid.shape))).cpu().float()

    start = timeit.default_timer()
    A = sitk.GetArrayFromImage(sitk.ReadImage(opt.fixed, sitk.sitkFloat32))
    B = sitk.GetArrayFromImage(sitk.ReadImage(opt.moving, sitk.sitkFloat32))

    #A = Variable(torch.from_numpy(A)).cuda(device).float()
    #B = Variable(torch.from_numpy(B)).cuda(device).float()

    # A, B = padding(A,B)
    #A = load_5D(A).cuda(device).float()
    #B = load_5D(B).cuda(device).float()

    A = load_5D(A)
    B = load_5D(B)

    A = Variable(torch.from_numpy(A)).cuda(device).float()
    B = Variable(torch.from_numpy(B)).cuda(device).float()

   #A = Variable(torch.from_numpy( load_5D(opt.fixed))).cuda(device).float()
   #B = Variable(torch.from_numpy( load_5D(opt.moving))).cuda(device).float()
    start2 = timeit.default_timer()
    print('Time for loading data: ', start2 - start) 
    pred = model(A,B)
    F_AB = pred.permute(0,2,3,4,1).data.cpu().numpy()[0, :, :, :, :]
    #F_AB = pred.permute(0,2,3,4,1).data.cuda(device).numpy()[0, :, :, :, :]

    F_AB = F_AB.astype(np.float32)*range_flow
    warped_A = transform(A,pred.permute(0,2,3,4,1)*range_flow,grid).data.cpu().numpy()[0, 0, :, :, :]
    #warped_A = transform(A,pred.permute(0,2,3,4,1)*range_flow,grid).data.cuda(device).numpy()[0, 0, :, :, :]
    start3 = timeit.default_timer()
    print('Time for registration: ', start3 - start2)
    warped_F_BA = transform(-pred,pred.permute(0,2,3,4,1)*range_flow,grid).permute(0,2,3,4,1).data.cpu().numpy()[0, :, :, :, :]
    #warped_F_BA = transform(-pred,pred.permute(0,2,3,4,1)*range_flow,grid).permute(0,2,3,4,1).data.cuda(device).numpy()[0, :, :, :, :]
    warped_F_BA = warped_F_BA.astype(np.float32)*range_flow
    start4 = timeit.default_timer()    
    print('Time for generating inverse flow: ', start4 - start3)
    save_flow(F_AB,savepath+'/flow_A_B.nii.gz')      
    save_flow(warped_F_BA,savepath+'/inverse_flow_B_A.nii.gz')   
    save_img(warped_A,savepath+'/warped_A.nii.gz')
    start5 = timeit.default_timer()    
    print('Time for saving results: ', start5 - start4)         
    del pred
    pred = model(B,A)
    F_BA = pred.permute(0,2,3,4,1).data.cpu().numpy()[0, :, :, :, :]
    #F_BA = pred.permute(0,2,3,4,1).data.cuda(device).numpy()[0, :, :, :, :]
    F_BA = F_BA.astype(np.float32)*range_flow
    warped_B = transform(B,pred.permute(0,2,3,4,1)*range_flow,grid).data.cpu().numpy()[0, 0, :, :, :]
    #warped_B = transform(B,pred.permute(0,2,3,4,1)*range_flow,grid).data.cuda(device).numpy()[0, 0, :, :, :]
    warped_F_AB = transform(-pred,pred.permute(0,2,3,4,1)*range_flow,grid).permute(0,2,3,4,1).data.cpu().numpy()[0, :, :, :, :]
    #warped_F_AB = transform(-pred,pred.permute(0,2,3,4,1)*range_flow,grid).permute(0,2,3,4,1).data.cuda(device).numpy()[0, :, :, :, :]
    warped_F_AB = warped_F_AB.astype(np.float32)*range_flow
    save_flow(F_BA,savepath+'/flow_B_A.nii.gz')      
    save_flow(warped_F_AB,savepath+'/inverse_flow_A_B.nii.gz')   
    save_img(warped_B,savepath+'/warped_B.nii.gz')


#imgshape = (144, 192, 160)
imgshape = (128, 128, 128)
range_flow = 7
test()
    
    
    
    
