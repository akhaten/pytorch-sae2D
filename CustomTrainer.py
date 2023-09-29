import pathlib
import torch
import torch.nn
import torch.nn.utils
import torch.optim
import torch.optim.lr_scheduler

import ignite.engine
import numpy
import typing

import wrapper2D.defineme

class CustomEngine(ignite.engine.Engine):

    def __init__(self, 
        process_function: typing.Callable[[ignite.engine.Engine, typing.Any], typing.Any]
    ):

        super(CustomEngine, self).__init__(process_function)

        # Add attribute for compute epoch loss
        self.epoch_loss = 0.
        self.counter = 0

        # Epoch loss history
        self.epoch_loss_history = []

def create_train_step(
    model: wrapper2D.defineme.SegmentationAutoEncoder,
    model_device: torch.device,
    datas_device: torch.device,
    optimizer: torch.optim.Optimizer, 
    criterion: wrapper2D.defineme.SAELoss2D,
    lr_scheduler: torch.optim.lr_scheduler.StepLR = None
):

    # Define any training logic for iteration update
    def train_step(engine: ignite.engine.Engine, batch):
        
        inputs, proba_map = batch[0], batch[1]
        
        # Move batch on model_device    
        # inputs = inputs.to(model_device, non_blocking=True)
        # results = results.to(model_device, non_blocking=True)
        
        batch_loss = 0.0
        size_of_batch = len(inputs)
        # acc_logits = []
        # acc_recon = []

        # Batch processing
        model.train()
        optimizer.zero_grad()
        for i in range(0, size_of_batch):
            print('Epoch: {}; Iter : [{}/{}]'.format(engine.state.epoch, i, size_of_batch))
            # to model device
            ## x : torch.Size([1, nb_channels, nb_rows, nb_cols])
            ## x_template : torch.Size([1, nb_classes, nb_rows, nb_cols])
            x = inputs[i].to(model_device, non_blocking=True)
            x_template = proba_map[i].to(model_device, non_blocking=True)

            recon, logits = model(x=x, prior=x_template, return_logits = True)
            # logits : torch.Size([1, nb_classes, nb_rows, nb_cols])
            # recon : torch.Size([1, nb_channels, nb_rows, nb_cols])
            loss: torch.Tensor = criterion(x=x, proba_map=x_template, logits=logits, recon=recon)
            batch_loss += loss.item()
            loss.backward()
      
            # acc_recon.append(recon.to(datas_device))
            # acc_logits.append(logits.to(datas_device))

            # return on datas device
            inputs[i] = x.to(datas_device, non_blocking=True)
            proba_map[i] = x_template.to(datas_device, non_blocking=True)

        batch_loss /= size_of_batch
        optimizer.step()
        # End batch processing

        
        # Dont use
        # if not(lr_scheduler is None):
        #     lr_scheduler.step()

        
        output = {
            # 'inputs' : inputs,
            # 'logits' : acc_logits,
            # 'recons' : acc_recon,
            'batch_loss' : batch_loss
        }

        
        return output

    return train_step



def clean_saeloss(engine: ignite.engine.Engine, loss: wrapper2D.defineme.SAELoss2D):
    loss.clear_running_var()


def update_epoch_loss(engine: CustomEngine) -> None:
    # Update batch loss
    bl = engine.state.output['batch_loss']
    engine.epoch_loss += bl
    # Update counter
    engine.counter += 1
    
def compute_epoch_loss(engine: CustomEngine) -> None:
    # Epoch loss is mean of all batch loss
    engine.epoch_loss /= engine.counter
    engine.epoch_loss_history.append(engine.epoch_loss)
    # Reset loss
    engine.epoch_loss = 0.
    engine.counter = 0

def save_epoch_loss(engine: CustomEngine, path: pathlib.Path) -> None:
    numpy.save(
        file = path,
        arr = numpy.array(engine.epoch_loss_history)
    )

def print_logs(engine: ignite.engine.Engine):
    strp = 'Epoch [{}/{}] : Loss {:.6f}'
    print(
        strp.format(
            engine.state.epoch,
            engine.state.max_epochs,
            engine.state.output['loss']
        )
    )

def save_model(
    engine: ignite.engine.Engine, 
    model: torch.nn.Module, 
    path: pathlib.Path = pathlib.Path('.')
) -> None:
    no_epoch = engine.state.epoch
    torch.save(model.state_dict(), path / 'model_epoch_{}.pt'.format(no_epoch))