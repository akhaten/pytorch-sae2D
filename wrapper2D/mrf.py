import torch
import torch.nn.functional

import numpy
import skimage


import sae.functions.mrf
import sae.functions.training_tools
import wrapper2D.training_tools

def get_lookup(prior, neighboor_size):
    '''
    prior = one-hot encoded prior segmentation as torch tensor
            shape should be [1, labels, dim1, dim2, dim3]
    '''
    k = neighboor_size
    assert prior.dtype == torch.uint8, 'The prior should be one-hot encoded byte'
    assert numpy.any(numpy.linspace(3,21,10) == k), ('Make sure that the neighboour_size' + 
                                               'is within numpy.linspace(3,21,10)')
    labels = prior.size(1)
    # print('Nb labels:', labels)
    enumerate_chs = torch.arange(labels).view(1, -1, 1, 1).byte()
    enumerate_chs = enumerate_chs.to(prior.device)
    
    print('prior device:', prior.device)
    print('enumerate_chs device:', enumerate_chs.device)
    enumerated_prior = enumerate_chs*prior
    enumerated_prior = torch.sum(enumerated_prior,1,True) 
    enumerated_prior = wrapper2D.training_tools.padder(
        enumerated_prior.float(), 
        k
    )
    enumerated_prior = enumerated_prior.squeeze().cpu().int().numpy()
    
    windows = skimage.util.view_as_windows(enumerated_prior, (k,k)) # windows.shape                                                                #(dim1, dim2, k, k)
    centers = windows[:, :, k//2, k//2]

    windows = windows.reshape(-1,k,k)
    centers = centers.reshape(-1)

    lookup_table = numpy.zeros((labels, labels))
    for condition in range(labels):
        idx = (centers == condition)   
        for s in range(labels):
            if condition == s:
                # removing repeated counts that comes from the center
                counts = numpy.sum(windows[idx] == s) - windows[idx].shape[0]
            else: 
                counts = numpy.sum(windows[idx] == s)
            lookup_table[condition, s] = counts

    norm = numpy.sum(lookup_table,
                  axis=1,
                  keepdims=True)
    norm = numpy.tile(norm, (1, labels))

    lookup_table = lookup_table/norm 

    assert (numpy.allclose(lookup_table.sum(1), 
                        numpy.ones_like(lookup_table.sum(1)))), 'Row doesnt add up to 1'
    
    return lookup_table


def neighboor_q(inumpyut, neighboor_size):
    '''
    Calculate the product of all q(s|x) around voxel i
    Uses convolution to sum the log(q_y) then takes the exp
    
    inumpyut: prob of q
    '''
    
    k = neighboor_size
    assert numpy.any(numpy.linspace(3,21,10) == k), ('Make sure that the neighboour_size' + 
                                            'is within numpy.linspace(3,21,10)')    
    x = wrapper2D.training_tools.padder(inumpyut,
                kernel_size= k)

    chs = x.shape[1]

    filter = torch.ones(k, k).view(1, 1, k, k)
    filter[:, :, k//2, k//2] = 0
    filter = filter.repeat(chs,1,1,1).float().to(inumpyut.device)
    filter.requires_grad = False

    out = torch.nn.functional.conv2d(x, 
                weight= filter,
                stride= 1, 
                groups= chs)
    return out
    

def spatial_consistency(inumpyut, table, neighboor_size):
    '''
    KL divergence between q(s|x) and markov random field
    
    inumpyut: prob of q
    table: lookup table as probability. Rows add up to 1
    '''
    eps = 1e-12
    n_batch, chs, dim1, dim2 = inumpyut.shape
    q_i = inumpyut 
    q_y = neighboor_q(inumpyut, neighboor_size)
    assert q_i.shape == q_y.shape, 'q_y and q_i should be the same shape'
    
    # To log probability table
    assert (numpy.allclose(table.sum(1), 
                        numpy.ones_like(table.sum(1)))), 'Row doesnt add up to 1'
    
    M = torch.from_numpy(table+eps).float()
    M = M/torch.sum(M, 1, True) #Normalize to account for the extra eps
    M = torch.log(M).to(inumpyut.device)
    assert M.shape == torch.Size([chs,chs]), 'Table dims dont match number of labels'
    M = M.view(1, chs, chs)
    
    #Multiplication
    q_i = inumpyut.view(n_batch, chs, dim1*dim2) 
    q_y = q_y.view(n_batch, chs, dim1*dim2)
    out = torch.bmm(M, q_y)  # shape [n_batch, chs, dim1*dim2*dim3]
    out = torch.sum(q_i*out,1)
    return -1*torch.sum(out)
