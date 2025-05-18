from torch import nn
from torch.autograd import grad
import torch
import pdb

class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True,  stride = 1, bias = True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias = bias)

    def forward(self, input):
        output = self.conv(input)
        return output

class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        return output

class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = input
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        output = self.conv(output)
        return output

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size,input_height,output_width,output_depth) for t_t in spl]
        output = torch.stack(stacks,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size,output_height,output_width,output_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=64):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'down':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'up':
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample == None:
            #TODO: ????
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size = kernel_size)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = UpSampleConv(input_dim, output_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(output_dim, output_dim, kernel_size = kernel_size)
        elif resample == None:
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(input_dim, output_dim, kernel_size = kernel_size)
        else:
            raise Exception('invalid resample value')

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output

"""
class GoodGenerator(nn.Module):
    def __init__(self, dim=64):
        super(GoodGenerator, self).__init__()
        self.dim = dim

        # Example: simple encoder-decoder structure for denoising
        self.enc1 = nn.Conv2d(3, dim, 3, padding=1)
        self.enc2 = nn.Conv2d(dim, 2*dim, 3, padding=1)
        self.enc3 = nn.Conv2d(2*dim, 4*dim, 3, padding=1)
        self.enc4 = nn.Conv2d(4*dim, 8*dim, 3, padding=1)

        self.dec1 = nn.ConvTranspose2d(8*dim, 4*dim, 3, stride=1, padding=1)
        self.dec2 = nn.ConvTranspose2d(4*dim, 2*dim, 3, stride=1, padding=1)
        self.dec3 = nn.ConvTranspose2d(2*dim, dim, 3, stride=1, padding=1)
        self.dec4 = nn.Conv2d(dim, 3, 3, padding=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        e4 = self.relu(self.enc4(e3))
        # Decoder
        d1 = self.relu(self.dec1(e4))
        d2 = self.relu(self.dec2(d1))
        d3 = self.relu(self.dec3(d2))
        out = self.tanh(self.dec4(d3))
        return out

class GoodDiscriminator(nn.Module):
    def __init__(self, dim=64):
        super(GoodDiscriminator, self).__init__()

        self.dim = dim

        self.ssize = self.dim // 16
        self.conv1 = MyConvo2d(3, self.dim, 3, he_init = False)
        self.rb1 = ResidualBlock(self.dim, 2*self.dim, 3, resample = 'down', hw=self.dim)
        self.rb2 = ResidualBlock(2*self.dim, 4*self.dim, 3, resample = 'down', hw=int(self.dim/2))
        self.rb3 = ResidualBlock(4*self.dim, 8*self.dim, 3, resample = 'down', hw=int(self.dim/4))
        self.rb4 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'down', hw=int(self.dim/8))
        self.ln1 = nn.Linear(self.ssize*self.ssize*8*self.dim, 1)

    def forward(self, input):
        output = input.contiguous()
        output = output.view(-1, 3, self.dim, self.dim)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = output.view(-1, self.ssize*self.ssize*8*self.dim)
        output = self.ln1(output)
        output = output.view(-1)
        return output
"""


class GoodDiscriminator(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        # Input: (batch, 3, 32, 32)
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),   # (batch, 32, 16, 16)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),  # (batch, 64, 8, 8)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),                               # (batch, 64*8*8)
            nn.Linear(64*8*8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.main(x)
    
class GoodGenerator(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        # Input: (batch, 3, 32, 32)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(3*32*32, 4*4*128),
            nn.BatchNorm1d(4*4*128),
            nn.ReLU(True)
        )
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),                         # (batch, 128, 4, 4)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # (batch, 64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # (batch, 32, 16, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # (batch, 3, 32, 32)
            nn.Tanh()  # Use Tanh if your images are normalized to [-1, 1]
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        x = self.deconv(x)
        return x