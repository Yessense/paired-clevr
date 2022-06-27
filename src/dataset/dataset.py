import json
import os

import numpy as np
import torch
import torchvision.io.image
from torch.utils.data import Dataset
from torchvision.io import read_image
import matplotlib.pyplot as plt


class PairedClevr(Dataset):
    def __init__(self, scenes_dir, img_dir, indices, with_labels=False):
        self.with_labels = with_labels
        self.scenes_dir = scenes_dir
        self.img_dir = img_dir
        self.img_template = "{name}_{idx:06d}.png"
        self.json_template = "scene_{idx:06d}.json"
        self._size = len(os.listdir(self.img_dir)) // 2
        self.features = ['shape', 'color', 'size', 'material', 'x', 'y']
        self.features_size: int = 6
        self.indices = indices

    @property
    def size(self) -> int:
        return self._size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]

        item = []
        for name in ['img', 'pair']:
            img_path = os.path.join(self.img_dir, self.img_template.format(name=name, idx=idx))
            img = read_image(img_path, mode=torchvision.io.image.ImageReadMode.RGB) / 255
            item.append(img)

        json_path = os.path.join(self.scenes_dir, self.json_template.format(idx=idx))
        with open(json_path) as json_file:
            annotations = json.load(json_file)

        exchange_labels = torch.zeros(self.features_size, dtype=bool)

        obj1 = annotations[0]['objects'][0]
        obj2 = annotations[1]['objects'][0]

        if obj1['shape'] != obj2['shape']:
            exchange_labels[0] = True
        elif obj1['color'] != obj2['color']:
            exchange_labels[1] = True
        elif obj1['size'] != obj2['size']:
            exchange_labels[2] = True
        elif obj1['material'] != obj2['material']:
            exchange_labels[3] = True
        elif obj1['3d_coords'][0] != obj2['3d_coords'][0]:
            exchange_labels[4] = True
        elif obj1['3d_coords'][1] != obj2['3d_coords'][1]:
            exchange_labels[5] = True
        else:
            raise ValueError(f'All features are the same {obj1}, {obj2}')
        item.append(exchange_labels.unsqueeze(-1))

        if self.with_labels == True:
            labels = []
            labels.append(obj1['shape'])
            labels.append(obj1['color'])
            labels.append(obj1['size'])
            labels.append(obj1['material'])
            labels.append(obj1['3d_coords'][0])
            labels.append(obj1['3d_coords'][1])
            item.append(labels)

        return item


if __name__ == '__main__':
    dataset = PairedClevr(scenes_dir='./dataset/data/scenes',
                          img_dir='./dataset/data/images', indices=list(range(10000)))
    start_idx = 300
    n_images = 2

    plt.figure(figsize=(30, 20))
    fig, ax = plt.subplots(n_images, 2)
    for i in range(n_images):
        img, pair, exchange_label = dataset[i + start_idx]
        print(*[dataset.features[i] for i, exchange in enumerate(exchange_label) if exchange])
        ax[i, 0].imshow(img.permute(1, 2, 0))
        ax[i, 1].imshow(pair.permute(1, 2, 0))
    plt.show()
