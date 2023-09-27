import torch
import torch.nn.functional
import torch.utils.data
import torchvision

import numpy

import os
import pathlib

import PIL.Image

import Utils


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, data_path: pathlib.Path, nb_classes: int, device: torch.device) -> None:
        """
        Args:
            root_dir (pathlib.Path): Directory with all the images.
        """
        super(ImageDataset, self).__init__()
        
        data_path: pathlib.Path = data_path
            
        input_path = data_path / 'input'

        self.items: list[torch.Tensor] = []


        for img_input_path in input_path.iterdir():

            #filename = img_input_path.name #filename with extension file
            filename = img_input_path.stem #filename without extension file
            
            # Load imgs
            img_npy = numpy.array(numpy.load(img_input_path))

            # Make proba map
            proba_map = Utils.create_probability_map(
                img = img_npy,
                k = nb_classes
            )

            proba_map = torch.tensor(proba_map)
            proba_map = proba_map.to(device, dtype=torch.float)
            proba_map = proba_map.unsqueeze(0)
            

            img_input = torch.tensor(img_npy)
            
            # Move on datas device
            img_input = img_input.to(device, dtype=torch.float)
            img_input = img_input.unsqueeze(0)
            img_input = img_input.unsqueeze(0)
            
            self.items.append((img_input, proba_map, filename))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, str]:
        return self.items[index]
    

def split_dataset(dataset: torch.utils.data.Dataset, train_size: float) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    n = len(dataset)
    train_n = int(train_size*n)
    test_n = n-train_n
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_n, test_n])
    return train_dataset, test_dataset


def get_batch_with_variable_size_image(batch):

    imgs_input = []
    imgs_ground_truth = []
    imgs_filename = []

    for elem in batch:
        imgs_input.append(elem[0])
        imgs_ground_truth.append(elem[1])
        imgs_filename.append(elem[2])

   
    # Your custom processing here
    return imgs_input, imgs_ground_truth, imgs_filename