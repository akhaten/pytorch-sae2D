import torch
import torch.nn.functional

import numpy

import sae.functions.training_tools

import wrapper2D.mrf

class SAELoss2D:

    """
    Warnings:
        - Running var must be clear to each epoch's end with clear_running_var()
    """

    def __init__(self,
        sigma: float,
        alpha: float = 1.0, 
        beta: float = 0.01, 
        eps: float = 1e-12,
        k: int = 3,
        var: float = 1e8
    ) -> None:
        
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.eps = eps
        self.k = k

        self.var = var

        # self.lookup = None
        # if self.beta != 0:
        #     argm_ch = sae.functions.visualization.argmax_ch(self.prior)
        #     argm_ch = argm_ch.type(torch.uint8)
        #     self.lookup = sae.functions.mrf.get_lookup(
        #         prior = argm_ch,
        #         neighboor_size = self.k
        #     )

        self.running_var = []

    def __call__(self,
        x: torch.Tensor,
        proba_map: torch.Tensor,
        logits: torch.Tensor,
        recon: torch.Tensor
    ) -> torch.Tensor:
        
 
        prior = proba_map # ie template
        
        log_prior = torch.log(
            sae.functions.training_tools.normalize_dim1(
                prior+self.eps
            )
        ).detach()

        lookup = None
        if self.beta != 0:
            argm_ch = self.index_likely_probable(prior)
            argm_ch = argm_ch.type(torch.uint8)
            # print(argm_ch)
            lookup = wrapper2D.mrf.get_lookup(
                prior = argm_ch,
                neighboor_size = self.k
            )
            
        
        
        prior_loss = self.compute_prior_loss(logits, log_prior)
        recon_loss = self.compute_recon_loss(x, recon)
        consistent = self.compute_consistent(logits, lookup)

        return prior_loss + recon_loss + consistent
    
    def compute_prior_loss(self, 
        logits: torch.Tensor,
        log_prior: torch.Tensor
    ) -> torch.Tensor:

        log_pi = torch.nn.functional.log_softmax(logits, 1)
        pi = torch.exp(log_pi)
        
        cce = -1*torch.sum(pi*log_prior,1)      #cross entropy
        cce = torch.sum(cce,(1,2))            #cce over all the dims
        cce = cce.mean()               
            
        h = -1*torch.sum(pi*log_pi,1)
        h = torch.sum(h,(1,2))
        h = h.mean()

        prior_loss = cce - h

        return prior_loss
    
    def compute_consistent(self, 
        logits: torch.Tensor,
        lookup: torch.Tensor
    ) -> torch.Tensor:
        
        if self.beta != 0: # ie not(self.lookup is None)
            log_pi = torch.nn.functional.log_softmax(logits, 1)
            pi = torch.exp(log_pi)
            consistent = self.beta*wrapper2D.mrf.spatial_consistency(
                inumpyut = pi,
                table = lookup,
                neighboor_size = self.k
            )
        else:
            consistent = torch.zeros(1, device=logits.device)
        
        return consistent
    
    def compute_recon_loss(self, 
        x: torch.Tensor, 
        recon: torch.Tensor
    ) -> torch.Tensor:
        
        _, _, dim1, dim2 = x.size()
        
        if self.sigma == 0:
            
            mse = (recon-x.detach())**2  #mse
            mse = torch.sum(mse,(1,2))    #mse over all dims
            mse = mse.mean()                  #avarage over all batches
            recon_loss = self.alpha * mse 
        
        elif self.sigma == 2:

            # Estimated Variance
            mse = (recon-x.detach())**2
            self.running_var.append(mse.detach().mean().item())

            rounded_var = 10**numpy.round(numpy.log10(self.var))

            # Weight Reconstruction loss
            mse = numpy.clip(0.5*(1/(rounded_var)),0, 500) * mse
            mse = torch.sum(mse,(1,2))    #mse over all dims
            mse = mse.mean()                  #avarage over all batches

            # Since args.var is a scalar now, we need to account for
            # the fact that we doing log det of a matrix
            # Therefore, we multiply by the dimension of the image

            c = dim1*dim2 #chs is 1 for image

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

    def index_likely_probable(
        self,
        proba_map: torch.Tensor
    ) -> torch.Tensor:
        
        _, nb_classes, n, m = proba_map.size()
        idx_lp = torch.zeros_like(proba_map, dtype=torch.bool)

        for i in range(0, n):
            for j in range(0, m):
                idx_lp[:, :, i, j] = torch.where(
                    proba_map[:, :, i, j] == proba_map[:, :, i, j].max(),
                    True,
                    False
                )

        return idx_lp
    

import wrapper2D.models
import wrapper2D.training_tools
import wrapper2D.mrf
# import sae.functions.models

class SegmentationAutoEncoder(torch.torch.nn.Module):

    def __init__(self, 
        in_channels: int,
        out_channels: int, 
        latent_dim: int,
        tau: float = 2/3
    ) -> None:

        """
        Params:
            - in_channels : nb_channels of image to segmentation
            - out_channels : nb_channels of segmented image
            - latent_dim : ch
        """
        
        super(SegmentationAutoEncoder, self).__init__()
        
        # Encoder
        enc_nf = [4, 8, 16, 32]
        dec_nf = [32, 16, 8, 4]
        # self.encoder = sae.functions.models.Simple_Unet(
        self.encoder = wrapper2D.models.Simple_Unet(
            input_ch = in_channels,
            out_ch = latent_dim,
            use_bn = False,
            enc_nf = enc_nf,
            dec_nf = dec_nf
        )

        # summary = torch.load(
        #     f = './weights/pretrained_encoder.pth.tar',
        #     map_location=torch.device('cpu')
        # )                        
        # _ = self.encoder.load_state_dict(
        #     summary['u1']
        # ) 

        # Decoder
        # self.decoder = sae.functions.models.Simple_Decoder(
        self.decoder = wrapper2D.models.Simple_Decoder(
            input_ch = latent_dim,
            out_ch = out_channels 
        )

        self.tau = tau


    def forward(self, 
        x: torch.Tensor, 
        prior: torch.Tensor, 
        tau: float = 2/3,
        return_logits: bool = True
    ) -> torch.Tensor:

        out = self.encoder(x)
        # out = functions.models.enforcer(prior, out)
        out = wrapper2D.models.enforcer(prior, out)
        n_batch, chs, dim1, dim2 = out.size()
        logits = out
        out = out.permute(0, 2, 3, 1)
        out = out.view(n_batch, dim1*dim2, chs)
        # pred = functions.training_tools.gumbel_softmax(out, tau)
        pred = wrapper2D.training_tools.gumbel_softmax(out, self.tau)
        pred = pred.view(n_batch, dim1, dim2, chs)
        pred = pred.permute(0, 3, 1, 2)

        recon = self.decoder(pred)

        return recon, logits if return_logits else recon
        # return logits, recon if self.training else recon