import random
import sys
from argparse import ArgumentParser
from typing import Union, List, Tuple, Optional
import numpy as np

import torch
import wandb
from matplotlib import pyplot as plt  # type: ignore
from torch.utils.data import DataLoader, Dataset

sys.path.append("..")
from src.dataset.dataset import PairedClevr  # type: ignore
from src.model.scene_vae import ClevrVAE  # type: ignore

from sklearn.manifold import TSNE  # type: ignore
import seaborn as sns  # type: ignore

parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')
program_parser.add_argument("--checkpoint_path", type=str,
                            default='/home/yessense/projects/paired-dsprites/src/model/checkpoints/epoch=290-step=57035.ckpt')
program_parser.add_argument("--batch_size", type=int, default=1)

# parse input
args = parser.parse_args()


class Stats:
    def __init__(self, checkpoint_path: str, allowed_indices=None):
        wandb.init(project="paired-clever-stats")
        if allowed_indices is None:
            allowed_indices = list(range(10000))
        self.device = 'cuda:0'
        self.model = self.load_model_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.allowed_indices = allowed_indices
        self.dataset = PairedClevr(scenes_dir='./dataset/data/scenes',
                                   img_dir='./dataset/data/images', indices=self.allowed_indices, with_labels=True)
        self.loader = DataLoader(self.dataset, batch_size=args.batch_size, num_workers=1, shuffle=True)

    def load_model_from_checkpoint(self, checkpoint_path: str) -> ClevrVAE:
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        hyperparams = ckpt['hyper_parameters']
        state_dict = ckpt['state_dict']

        model = ClevrVAE(**hyperparams)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    def encode_vectors(self, batch_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """Encode all images in dataset"""
        features_list: List[torch.Tensor] = []
        labels_list: List = []
        dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size)

        for image1, _, _, labels in dataloader:

            image1 = image1.to(self.model.device)
            features = self.model.encoder(image1)
            features = self.model.reparameterize(*features)
            features = features.view(-1, self.model.n_features, self.model.latent_dim)
            features = features.cpu().detach().numpy()
            features_list.append(features)

            print(len(labels))
            print(len(labels[0]))

            labels_list.append(labels)


        return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)

    def visualize_feature(self, features: np.ndarray, labels: np.ndarray, n_feature: int, title: Optional[int] = None):
        """Visualize concrete feature specified by `n_feature`"""
        if title is None:
            title = n_feature

        print(features.shape)
        print(labels.shape)
        n_components = 2
        tsne = TSNE(n_components, learning_rate=1.2, init='pca', random_state=42)
        tsne_result = tsne.fit_transform(features[:, n_feature])
        labels = labels[:, title]

        sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels)
        plt.title(f'TSNE visualization of {self.dataset.features[n_feature]!r}')
        plt.legend(title=self.dataset.features[title])
        wandb.log({"Visualize feature": plt})

    def visualize_objets(self, features: np.ndarray, labels: np.ndarray):
        """Visualize sum of features colored by feature"""
        n_components = 2
        tsne = TSNE(n_components, learning_rate=200, init='random', random_state=42)
        tsne_result = tsne.fit_transform(np.sum(features, axis=1))

        for title in range(6):
            sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels[:, title])
            plt.title(f'TSNE visualization of object vectors colored by feature')
            plt.legend(title=f'{self.dataset.features[title]}')
            wandb.log({"Visualize sum of features": plt})

    # def show_feature_mean_vectors(self, features: np.ndarray, labels: np.ndarray, feature: int):
    #     vectors = np.sum(features, axis=1)
    #     mean_vectors = []
    #
    #     for i in range(self.dataset.features_size[feature]):
    #         mean_vectors.append(np.mean(vectors[labels[:, feature] == i], axis=0))
    #
    #     fig, ax = plt.subplots(figsize=(30, 8))
    #     mean_vectors = np.array(mean_vectors)
    #     sns.heatmap(mean_vectors, ax=ax, center=0)
    #     feature_name = self.dataset.features[feature]
    #     fig.suptitle(f'Vector means at same position for feature {feature_name}')
    #     ax.set_ylabel(f'{feature_name}')
    #     ax.set_xlabel('coordinate')
    #     plt.show()


if __name__ == '__main__':
    n_features = 6

    stats = Stats(args.checkpoint_path, allowed_indices=random.sample(list(range(10000)), 1000))

    features, labels = stats.encode_vectors(128)

    for i in range(n_features):
        stats.visualize_feature(n_feature=i, features=features, labels=labels, title=1)
    stats.visualize_objets(features, labels)

    print("Done")
