import torch.autograd
import torch
import torch.nn.functional
import pathlib


import wrapper.training_tools
import wrapper.models


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

    @staticmethod 
    def forward(self, prior, x):
        # Regions that prior have 0 prob, we dont want to sample from it
        # Adding a very large negative value to the logits (log of the unormalized prob)
        # hopefully prevent that regions from being sample
        eps = 1e-8
        self.forbidden_regs = (prior < eps).float()       
        return x - 1e12*self.forbidden_regs

    @staticmethod
    def backward(self, grad):
        # Make sure that the forbidden regions have 0 gradiants.
        return grad*(self.forbidden_regs==0).float()

def enforcer(prior, x):
    enforce_prior = EnforcePrior(prior)
    to_apply = enforce_prior.apply
    return to_apply(prior, x)




import torch
import torch.nn.functional
import torch.optim

import numpy

import sae.functions.mrf
import sae.functions.visualization
import sae.functions.training_tools

import wrapper.mrf



class SAELoss:

    """
    Warnings:
        - Running var must be clear to each epoch's end with clear_running_var()
    """

    def __init__(self,
        sigma: float,
        prior,
        alpha: float = 1.0, 
        beta: float = 0.01, 
        eps: float = 1e-12,
        k: int = 3,
        var: float = 1e8
    ) -> None:
        
        self.prior = prior
        self.log_prior = torch.log(
            sae.functions.training_tools.normalize_dim1(prior+eps)
        ).detach()
        
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.eps = eps
        self.k = k

        self.var = var

        self.lookup = None
        if self.beta != 0:
            argm_ch = sae.functions.visualization.argmax_ch(self.prior)
            argm_ch = argm_ch.type(torch.uint8)
            self.lookup = sae.functions.mrf.get_lookup(
                prior = argm_ch,
                neighboor_size = self.k
            )

        self.running_var = []


    def __call__(self,
        x: torch.Tensor,
        logits: torch.Tensor,
        recon: torch.Tensor
    ) -> torch.Tensor:
        prior_loss = self.compute_prior_loss(logits)
        recon_loss = self.compute_recon_loss(x, recon)
        consistent = self.compute_consistent(logits)
        return prior_loss + recon_loss + consistent

    def compute_prior_loss(self, logits: torch.Tensor) -> torch.Tensor:

        log_pi = torch.nn.functional.log_softmax(logits, 1)
        pi = torch.exp(log_pi)
        
        cce = -1*torch.sum(pi*self.log_prior,1)      #cross entropy
        cce = torch.sum(cce,(1,2,3))            #cce over all the dims
        cce = cce.mean()               
            
        h = -1*torch.sum(pi*log_pi,1)
        h = torch.sum(h,(1,2,3))
        h = h.mean()
 
        prior_loss = cce - h

        return prior_loss
    
    def compute_consistent(self, logits: torch.Tensor) -> torch.Tensor:
        
        log_pi = torch.nn.functional.log_softmax(logits, 1)
        pi = torch.exp(log_pi)
        
        if self.beta != 0: # ie not(self.lookup is None)
            consistent = self.beta*wrapper.mrf.spatial_consistency(
                input = pi,
                table = self.lookup,
                neighboor_size = self.k
            )
        else:
            consistent = torch.zeros(1, device=logits.device)
        
        return consistent
    
    def compute_recon_loss(self, 
        x: torch.Tensor, 
        recon: torch.Tensor
    ) -> torch.Tensor:

        if self.sigma == 0:
            mse = (recon-x.detach())**2  #mse
            mse = torch.sum(mse,(1,2,3,4))    #mse over all dims
            mse = mse.mean()                  #avarage over all batches
            recon_loss = self.alpha * mse 
        elif self.sigma == 2:
            mse = (recon-x.detach())**2
            rounded_var = 10**numpy.round(numpy.log10(self.var))

            # Weight Reconstruction loss
            mse = numpy.clip(0.5*(1/(rounded_var)),0, 500) * mse
            mse = torch.sum(mse,(1,2,3,4))    #mse over all dims
            mse = mse.mean()                  #avarage over all batches

            self.running_var.append(mse.detach().mean().item())

            # Since args.var is a scalar now, we need to account for
            # the fact that we doing log det of a matrix
            # Therefore, we multiply by the dimension of the image

            c = dim1*dim2*dim3 #chs is 1 for image

            _var = torch.from_numpy(numpy.array(self.var+self.eps)).float()
            recon_loss = mse + 0.5 * c * torch.log(_var)
        else:
            raise AssertionError('sigma must be 0 or 2')
        
        return recon_loss
    
    def update_variance(self) -> None:
        self.var = numpy.mean(self.running_var)

    def clear_running_var(self) -> None:
        """ Running var must be clear to each end epoch
        """
        self.running_var.clear()    