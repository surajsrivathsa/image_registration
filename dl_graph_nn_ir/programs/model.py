import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


class Admir_Affine_Encoder(nn.Module):

    def __init__(self, in_channel, start_channel, num_conv_blocks=6):
        self.in_channel = in_channel
        self.start_channel = start_channel
        self.num_conv_blocks = num_conv_blocks
        self.encoder_layer_list = []
        super(Admir_Affine_Encoder, self).__init__()
        self.create_model()

    def affine_conv_block(self, in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, bias=True ):
      layer = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                            nn.BatchNorm3d(out_channels),
                            nn.LeakyReLU(negative_slope=0.1))
      return layer
    

    def create_model(self):
      for i in range(self.num_conv_blocks):
          if(i == 0):
            lyr = self.affine_conv_block(in_channels = self.in_channel, out_channels = self.start_channel)
            self.encoder_layer_list.append(lyr)
          else:
            lyr = self.affine_conv_block(in_channels= self.start_channel * i, out_channels = self.start_channel * (i+1))
            self.encoder_layer_list.append(lyr)

    def forward(self, x, y):

      # print("x,y", x.shape, "  ", y.shape)
      x_in=torch.cat((x, y), 1)
      e0 = self.encoder_layer_list[0](x_in)
      e1 = self.encoder_layer_list[1](e0)
      e2 = self.encoder_layer_list[2](e1)
      e3 = self.encoder_layer_list[3](e2)
      e4 = self.encoder_layer_list[4](e3)
      return e4


class Admir_Affine_Output(nn.Module):
  def __init__(self, in_units, out_units=128, dropout_prob = 0.3):
    
    self.in_units = in_units
    self.out_units = out_units
    self.dropout_prob = dropout_prob
    super(Admir_Affine_Output, self).__init__()
    self.trns_ob = self.translation_output_block(self.in_units, self.out_units)
    self.rss_ob = self.rot_scale_shear_output_block(self.in_units, self.out_units)
    return;
  
  def translation_output_block(self, in_units, out_units):
    layer = nn.Sequential(
          nn.Linear(in_features = in_units, out_features= out_units),
          nn.Dropout(p=self.dropout_prob),
          nn.Linear(in_features=out_units, out_features= out_units//2),
          nn.Dropout(p=self.dropout_prob),
          nn.Linear(in_features=out_units//2, out_features= out_units//4),
          nn.Dropout(p=self.dropout_prob),
          nn.Linear(in_features=out_units//4, out_features= out_units//8),
          nn.Dropout(p=self.dropout_prob),
          nn.Linear(in_features=out_units//8, out_features= 3))
    return layer

  def rot_scale_shear_output_block(self, in_units, out_units):
    layer = nn.Sequential(
          nn.Linear(in_features = in_units, out_features= out_units),
          nn.Dropout(p=self.dropout_prob),
          nn.Linear(in_features=out_units, out_features= out_units//2),
          nn.Dropout(p=self.dropout_prob),
          nn.Linear(in_features=out_units//2, out_features= out_units//4),
          nn.Dropout(p=self.dropout_prob),
          nn.Linear(in_features=out_units//4, out_features= out_units//8),
          nn.Dropout(p=self.dropout_prob),
          nn.Linear(in_features=out_units//8, out_features= 9),
          nn.Tanh())
    return layer
  
  def forward(self, input_tnsr):
    ip = input_tnsr.flatten(start_dim=1, end_dim=4)
    #print(ip.shape)
    translation_output = self.trns_ob(ip)
    rotate_scale_shear_output = self.rss_ob(ip)
    return [translation_output, rotate_scale_shear_output]


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, is_affine=False, theta = None, mode='bilinear'):
        super().__init__()

        self.mode = mode
        self.isaffine = is_affine
        self.theta = theta
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):      
      if (self.isaffine):
        grid = F.affine_grid(self.theta, (2, 1, 128, 128, 128))
        warped_image = F.grid_sample(src, grid)
        return warped_image
      else:
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class Admir_Deformable_UNet(nn.Module):
  def __init__(self,in_channel  , n_classes,start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        super(Admir_Deformable_UNet, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=False)

        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=False)
        self.ec2 = self.encoder(self.start_channel, self.start_channel*2, stride=2, bias=False)

        self.ec3 = self.encoder(self.start_channel*2, self.start_channel*2, bias=False)
        self.ec4 = self.encoder(self.start_channel*2, self.start_channel*4, stride=2, bias=False)

        self.ec5 = self.encoder(self.start_channel*4, self.start_channel*4, bias=False)
        self.ec6 = self.encoder(self.start_channel*4, self.start_channel*8, stride=2, bias=False)

       
    
        self.dc1 = self.encoder(self.start_channel*8, self.start_channel*8, kernel_size=3, stride=1, bias=False) 
        self.dc2 = self.encoder(self.start_channel*4, self.start_channel*4, kernel_size=3, stride=1, bias=False)          
        self.dc3 = self.encoder(self.start_channel*2, self.start_channel*2, kernel_size=3, stride=1, bias=False)

        self.up1 = self.decoder(self.start_channel*8, self.start_channel*4)
        self.up2 = self.decoder(self.start_channel*4, self.start_channel*2)
        self.up3 = self.decoder(self.start_channel*2, self.start_channel)

        self.dc4 = self.output(self.start_channel, self.n_classes,kernel_size=1,bias=False)

  def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True):
    layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(negative_slope=0.1))
    return layer

  def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
    layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.LeakyReLU(negative_slope=0.1))
    return layer
       
  def output(self, in_channels, out_channels, kernel_size=3, 
                bias=False, batchnorm=False):
    layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias),
               )
    return layer

  def forward(self, x,y):
        # print("x,y", x.shape, "  ", y.shape)
        x_in=torch.cat((x, y), 1)  
        e0 = self.eninput(x_in)

        # print("e0", e0.shape)

        e0 = self.ec1(e0)
        es1 = self.ec2(e0)   #strided
        # print("e0", e0.shape)
        # print("es1", es1.shape)

        e1 = self.ec3(es1)   
        es2 = self.ec4(e1)   #strided
        # print("e1", e1.shape)
        # print("es2", es2.shape)

        e2 = self.ec5(es2)
        es3 = self.ec6(e2)   #strided
        # print("e2", e2.shape)
        # print("es3", es3.shape)

        

        d0 = self.dc1(es3)
        # print("d0", d0.shape)

        d0 = torch.add(self.up1(d0), e2)
        # print("d0", d0.shape)

        d1 = self.dc2(d0)
        d1 = torch.add(self.up2(d1), e1)
        # print("d1", d1.shape)

        d2 = self.dc3(d1)
        d2 = torch.add(self.up3(d2), e0)
        print("d2", d2.shape)

        output = self.dc4(d2)
        return output

    














