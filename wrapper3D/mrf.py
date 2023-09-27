import torch
import torch.nn.functional

import numpy

import functions.mrf
import functions.training_tools


def neighboor_q(input, neighboor_size):
    '''
    Calculate the product of all q(s|x) around voxel i
    Uses convolution to sum the log(q_y) then takes the exp
    
    input: prob of q
    '''
    
    k = neighboor_size
    assert numpy.any(numpy.linspace(3,21,10) == k), ('Make sure that the neighboour_size' + 
                                            'is within np.linspace(3,21,10)')    
    x = functions.training_tools.padder(input,
                kernel_size= k)

    chs = x.shape[1]

    filter = torch.ones(k, k, k).view(1, 1, k, k, k)
    filter[:, :, k//2, k//2, k//2] = 0
    filter = filter.repeat(chs,1,1,1,1).float().to(input.device)
    filter.requires_grad = False

    out = torch.nn.functional.conv3d(x, 
                weight= filter,
                stride= 1, 
                groups= chs)
    return out
    

def spatial_consistency(input, table, neighboor_size):
    '''
    KL divergence between q(s|x) and markov random field
    
    input: prob of q
    table: lookup table as probability. Rows add up to 1
    '''
    eps = 1e-12
    n_batch, chs, dim1, dim2, dim3 = input.shape
    q_i = input 
    q_y = neighboor_q(input, neighboor_size)
    assert q_i.shape == q_y.shape, 'q_y and q_i should be the same shape'
    
    # To log probability table
    assert (numpy.allclose(table.sum(1), 
                        numpy.ones_like(table.sum(1)))), 'Row doesnt add up to 1'
    
    M = torch.from_numpy(table+eps).float()
    M = M/torch.sum(M, 1, True) #Normalize to account for the extra eps
    M = torch.log(M).to(input.device)
    assert M.shape == torch.Size([chs,chs]), 'Table dims dont match number of labels'
    M = M.view(1, chs, chs)
    
    #Multiplication
    q_i = input.view(n_batch, chs, dim1*dim2*dim3) 
    q_y = q_y.view(n_batch, chs, dim1*dim2*dim3)
    out = torch.bmm(M, q_y)  # shape [n_batch, chs, dim1*dim2*dim3]
    out = torch.sum(q_i*out,1)
    return -1*torch.sum(out)