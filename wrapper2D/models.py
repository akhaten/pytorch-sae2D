import torch.autograd
import torch
import torch.nn.functional


import numpy

import sae.functions.mrf
import sae.functions.visualization
import sae.functions.training_tools

import wrapper2D.models


class Simple_Unet(torch.nn.Module):

    def __init__(self, input_ch, out_ch, use_bn, enc_nf, dec_nf, ignore_last=False):
        super(Simple_Unet, self).__init__()
        
        self.ignore_last = ignore_last
        self.down = torch.nn.MaxPool2d(2,2)

        self.block0 = simple_block(input_ch , enc_nf[0], use_bn)
        self.block1 = simple_block(enc_nf[0], enc_nf[1], use_bn)
        self.block2 = simple_block(enc_nf[1], enc_nf[2], use_bn)
        self.block3 = simple_block(enc_nf[2], enc_nf[3], use_bn)

        self.block4 = simple_block(enc_nf[3], dec_nf[0], use_bn)    
        
        self.block5 = simple_block(dec_nf[0]*2, dec_nf[1], use_bn)       
        self.block6 = simple_block(dec_nf[1]*2, dec_nf[2], use_bn)         
        self.block7 = simple_block(dec_nf[2]*2, dec_nf[3], use_bn) 
        self.block8 = simple_block(dec_nf[3]*2, out_ch,    use_bn)           

        self.conv = torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x_in):

        #Model
        x0 = self.block0(x_in)
        x1 = self.block1(self.down(x0))
        x2 = self.block2(self.down(x1))
        x3 = self.block3(self.down(x2))

        x = self.block4(self.down(x3))
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        
        # if not x.size() == x3.size():                  
        #     x = torch.functional.pad(x,(0,1,0,0,0,1) , mode='replicate')
            
        x = torch.cat([x, x3], 1)
        x = self.block5(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
 
        x = torch.cat([x, x2], 1)
        x = self.block6(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        
        x = torch.cat([x, x1], 1)
        x = self.block7(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')

        x = torch.cat([x, x0], 1)
        x = self.block8(x)
        
        if self.ignore_last:
            out = x
        else:
            out = self.conv(x)
        
        return out    
    
class simple_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(simple_block, self).__init__()
        
        self.use_bn= use_bn
        
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.InstanceNorm2d(out_channels)
            
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.activation(out)
        return out
    
def Simple_Decoder(input_ch, out_ch):
    chs = [input_ch, out_ch]
    conv2d = torch.nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1)
    layers = [conv2d, torch.nn.Sigmoid()]
    return torch.nn.Sequential(*layers)



class EnforcePrior(torch.autograd.Function):
    """
    Check:
    - https://pytorch.org/docs/stable/autograd.html
    - https://pytorch.org/docs/stable/notes/extending.html#extending-autograd
    """

    @staticmethod 
    def forward(ctx, prior, x):
        # Regions that prior have 0 prob, we dont want to sample from it
        # Adding a very large negative value to the logits (log of the unormalized prob)
        # hopefully prevent that regions from being sample
        eps = 1e-8
        forbidden_regs = (prior < eps).float()    
        result = x - 1e12*forbidden_regs
        ctx.save_for_backward(forbidden_regs)
        return result

    # @staticmethod
    # def setup_context(ctx, inputs, output):
    #     _, x = inputs
    #     ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad):
        # Make sure that the forbidden regions have 0 gradiants.
        forbidden_regs, = ctx.saved_tensors

        grad_prior = None
        grad_x = grad*(forbidden_regs==0).float()

        return grad_prior, grad_x

def enforcer(prior, x):
    # Check https://pytorch.org/docs/stable/notes/extending.html#extending-autograd
    enforce_prior = EnforcePrior()
    to_apply = enforce_prior.apply
    return to_apply(prior, x)


