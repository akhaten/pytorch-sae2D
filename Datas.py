import torch
import torch.nn.functional
import torch.utils.data

import numpy

import pathlib


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, data_path: pathlib.Path, device: torch.device) -> None:
        """
        Args:
            root_dir (pathlib.Path): Directory with all the images.
        """
        super(ImageDataset, self).__init__()
        
        data_path: pathlib.Path = data_path
            
        input_path = data_path / 'input'
        proba_map_path = data_path / 'proba_map'

        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []


        zipped = zip(input_path.iterdir(), proba_map_path.iterdir())
        for img_input_path, proba_map_path in zipped:

            #filename = img_input_path.name #filename with extension file
            filename = img_input_path.stem #filename without extension file
            
            # Load imgs
            img_input = torch.tensor(numpy.load(img_input_path))
            img_input = img_input.unsqueeze(0).unsqueeze(0)
            img_proba_map = torch.tensor(numpy.load(proba_map_path))
            img_proba_map = img_proba_map.unsqueeze(0)

            # print(img_input.size(), img_proba_map.size())
            
            # Move on datas device
            img_input = img_input.to(device, dtype=torch.float)
            img_proba_map = img_proba_map.to(device, dtype=torch.float)
            
            self.items.append((img_input, img_proba_map, filename))


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
    imgs_proba_map = []
    imgs_filename = []

    for elem in batch:
        imgs_input.append(elem[0])
        imgs_proba_map.append(elem[1])
        imgs_filename.append(elem[2])

   
    # Your custom processing here
    return imgs_input, imgs_proba_map, imgs_filename