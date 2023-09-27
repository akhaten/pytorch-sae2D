import torch.nn
import torch.utils.data

import torchvision.utils
import numpy

import ignite.engine

import pathlib

#from Unfolding2D import \
#    Datas
import Datas

import Utils
import wrapper2D.defineme
        

def evaluate_dataloader(
    engine: ignite.engine.Engine,
    model: wrapper2D.defineme.SegmentationAutoEncoder,
    model_device: torch.device,
    datas_device: torch.device,
    dataloader: Datas.ImageDataset,
    path_eval: pathlib.Path
) -> None:
    
    model.eval() # model.train(False)
        
    with torch.no_grad():
        
        no_epoch = engine.state.epoch
        imgs_output_path = path_eval / 'epoch_{}'.format(no_epoch)

        if not(imgs_output_path.exists()):
            imgs_output_path.mkdir()

        for batch in dataloader:

            inputs, classes, filename = batch

            # Move batch on model_device
            inputs = inputs.to(model_device, non_blocking=True)

            for i in range(0, inputs.shape[0]):
                # to model device
                x = inputs[i].to(model_device, non_blocking=True)
                k = classes[i]
                # x : torch.Size([1, nb_channels, nb_rows, nb_cols])
                # proba_map = Utils.create_probability_map(
                #     img = x,
                #     k = k
                # )
                img = x[0, 0].cpu().detach().numpy()
                proba_map = Utils.create_probability_map(
                    img = img,
                    k = k.item()
                )
                proba_map = torch.tensor(
                    proba_map, 
                    device=model_device, 
                    dtype=x.dtype,
                    requires_grad = False
                )
                proba_map = proba_map.unsqueeze(0)
                recon = model(
                    x=x, 
                    prior=proba_map, 
                    return_logits = False
                )
                # logits : torch.Size([1, nb_classes, nb_rows, nb_cols])
                # recon : torch.Size([1, nb_channels, nb_rows, nb_cols])
                recon = recon.cpu()
                torchvision.utils.save_image(
                    recon[0, 0], 
                    imgs_output_path / (filename[i]+'.png')
                )
                numpy.save(
                    imgs_output_path / (filename[i]+'.npy'), 
                    recon[0, 0].detach().numpy()
                )

            # Batch return on datas device
            inputs = inputs.to(datas_device, non_blocking=True)

