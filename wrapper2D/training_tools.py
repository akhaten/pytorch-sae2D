import torch
import torch.nn.functional


# def dice_loss(pred, target, ign_first_ch=True):

#     eps = 1
#     assert pred.size() == target.size(), 'Input and target are different dim'
    
#     if len(target.size())==4:
#         n,c,x,y = target.size()
#     # if len(target.size())==5:
#     #     n,c,x,y,z = target.size()

#     target = target.view(n,c,-1)
#     pred = pred.view(n,c,-1)
    
#     if ign_first_ch:
#         target = target[:,1:,:]
#         pred = pred[:,1:,:]
 
#     num = torch.sum(2*(target*pred),2) + eps
#     den = (pred*pred).sum(2) + (target*target).sum(2) + eps
#     dice_loss = 1-num/den
#     ind_avg = dice_loss
#     total_avg = torch.mean(dice_loss)
#     regions_avg = torch.mean(dice_loss, 0)
    
#     return total_avg, regions_avg, ind_avg


def sample_gumbel(shape, device: torch.device, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits: torch.Tensor, temperature: float):
    y = logits + sample_gumbel(logits.size(), logits.device)
    return torch.nn.functional.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard


def padder(input, kernel_size):
    '''
    Use to pad input. 
    Mainly to produce the same size output after gaussian smoothing
    '''
    assert input.dtype == torch.float32, 'Input must be torch.float32'
    assert kernel_size>1, 'Gaussian-smoothing kernel must be greater than 1'
    
    p = (kernel_size+1)//2
    r = kernel_size%2 

    if r>0:
        input = torch.nn.functional.pad(
            input, 
            (p-r,p-r,p-r,p-r), 
            mode='replicate'
        )
    elif r==0:
        input = torch.nn.functional.pad(
            input, 
            (p-1, p, p-1, p), 
            mode='replicate'
        )
    return input