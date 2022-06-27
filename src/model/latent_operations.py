import os
import random
from argparse import ArgumentParser
from typing import Union, List

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import sys

sys.path.append("..")
from src.dataset.dataset import PairedClevr
from src.model.scene_vae import ClevrVAE
import wandb

parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')
program_parser.add_argument("--checkpoint_path", type=str,
                            default='/home/akorchemnyi/paired-clevr/src/paired-clevr/1voroobi/checkpoints/epoch=84-step=47854.ckpt')
program_parser.add_argument("--batch_size", type=int, default=1)
program_parser.add_argument("--one", type=int, default=1)
program_parser.add_argument("--several", type=int, default=1)
program_parser.add_argument("--random", type=int, default=1)
program_parser.add_argument("--decode", type=int, default=1)

# parse input
args = parser.parse_args()


class Experiment:
    def __init__(self, checkpoint_path: str, cuda=True):
        wandb.init(project="paired-clever-latent-operations")
        self.device = torch.device('cuda:0')
        self.model = self.load_model_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.dataset = PairedClevr(scenes_dir='./dataset/data/scenes',
                                   img_dir='./dataset/data/images', indices=list(range(10000)))
        self.loader = DataLoader(self.dataset, batch_size=args.batch_size, num_workers=1, shuffle=True)
        self.img_template = "{name}_{idx:06d}.png"

    def load_model_from_checkpoint(self, checkpoint_path: str) -> PairedClevr:
        ckpt = torch.load(checkpoint_path, map_location='cuda:0')

        hyperparams = ckpt['hyper_parameters']
        state_dict = ckpt['state_dict']

        model = ClevrVAE(**hyperparams)
        model.load_state_dict(state_dict)
        # torch.save(model.encoder.state_dict(), './checkpoint/encoder_state_dict.pt')

        return model

    def forward_images(self):
        batch = next(iter(self.loader))

        img1, img2, exchange_labels = batch
        latent1 = self.model.get_latent(img1)
        r1 = self.model.reconstruct(latent1)

        fig, ax = plt.subplots(args.batch_size, 2)
        for i in range(args.batch_size):
            img = img1[i]
            pair_img = r1[i]

            ax[i, 0].imshow(img.detach().cpu().numpy().squeeze(0), cmap='gray')
            ax[i, 0].set_axis_off()
            ax[i, 1].imshow(pair_img.detach().cpu().numpy().squeeze(0), cmap='gray')
            ax[i, 1].set_axis_off()

        plt.figure(figsize=(20, 8))
        plt.show()

    def exchange_feature(self, feat1, feat2, n_feature: Union[List[int]]):
        """ Exchange n-th feature between 2 sets of features"""

        exchange_label = torch.ones(1, 6, 1024)
        exchange_label[:, n_feature, :] = 0
        exchange_label = exchange_label.to(self.model.device).bool()

        out = torch.where(exchange_label, feat1, feat2)
        return out

    def swap_one_feature(self):
        """With given 2 images swap one feature at a time for all features"""
        fig, ax = plt.subplots(6, 5, figsize=(10, 10))
        batch1 = next(iter(self.loader))
        batch2 = next(iter(self.loader))

        img1, _img, exchange_labels = batch1
        img2, _img, _exchange_labels = batch2
        z1 = self.model.encode_features_inference(img1.to(self.device))
        z2 = self.model.encode_features_inference(img2.to(self.device))

        r1 = self.model.decoder(torch.sum(z1, dim=1))
        r2 = self.model.decoder(torch.sum(z2, dim=1))

        y_labels = ('shape', 'color', 'size', 'material', 'x', 'y')
        x_labels = ('Image 1', 'Reconstructed 1', 'Exchanged', 'Reconstructed 2', 'Image 2')

        for i in range(6):
            z_exch = self.exchange_feature(z1, z2, i)
            z_exch = torch.sum(z_exch, dim=1)

            r3 = self.model.decoder(z_exch)
            for j, img in enumerate([img1[0], r1[0], r3[0], r2[0], img2[0]]):
                ax[i, j].imshow(img.detach().cpu().numpy().transpose(1, 2, 0), cmap='gray')
                if j == 0:
                    ax[i, j].set_ylabel(y_labels[i])
                if i == 5:
                    ax[i, j].set_xlabel(x_labels[j])
                if j != 0 and i != 5:
                    ax[i, j].set_axis_off()
        fig.tight_layout()
        wandb.log({"Swap one feature": plt})

    def swap_several_features(self):
        """With given 2 images swap 0-5 features at a time"""
        fig, ax = plt.subplots(7, 5, figsize=(10, 10))
        batch1 = next(iter(self.loader))
        batch2 = next(iter(self.loader))
        names = ['Исходная сц. 1', 'Декодированная сц.1', 'Декодированная сц. 2', 'Исходная сц. 2']

        img1, _img, exchange_labels = batch1
        img2, _img, _exchange_labels = batch2
        z1 = self.model.encode_features_latent(img1.to(self.device))
        z2 = self.model.encode_features_latent(img2.to(self.device))

        r1 = self.model.decoder(torch.sum(z1, dim=1))
        r2 = self.model.decoder(torch.sum(z2, dim=1))

        y_labels = (
            'None', 'Shp', 'Shp, Color', 'Shp, Col, Size', 'Shp, Col, Sz, M', 'Shp, Col, Sz, M, X',
            'Shp, Col, Sz, M, X, Y')
        x_labels = ('Image 1', 'Reconstructed 1', 'Exchanged', 'Reconstructed 2', 'Image 2')

        for i in range(7):
            z_exch = self.exchange_feature(z1, z2, list(range(i)))

            z_exch = torch.sum(z_exch, dim=1)
            r3 = self.model.decoder(z_exch)
            for j, img in enumerate([img1[0], r1[0], r3[0], r2[0], img2[0]]):
                ax[i, j].imshow(img.detach().cpu().numpy().transpose(1, 2, 0), cmap='gray')
                if j == 0:
                    ax[i, j].set_ylabel(y_labels[i])
                if i == 5:
                    ax[i, j].set_xlabel(x_labels[j])
                if j != 0 and i != 5:
                    ax[i, j].set_axis_off()
        fig.tight_layout()
        wandb.log({"Swap several features": plt})

    def swap_random_features(self):
        short_y_labels = {0: 'Shp', 1: 'Col', 2: 'Size', 3: 'Material', 4: 'X', 5: 'Y'}

        def make_y_label_name(idx):
            names = [short_y_labels[i] for i in idx]
            name = ", ".join(names)
            return name

        fig, ax = plt.subplots(6, 5, figsize=(10, 10))

        batch1 = next(iter(self.loader))
        batch2 = next(iter(self.loader))

        img1, _img, exchange_labels = batch1
        img2, _img, _exchange_labels = batch2
        z1 = self.model.encode_features_inference(img1.to(self.device))
        z2 = self.model.encode_features_inference(img2.to(self.device))

        r1 = self.model.decoder(torch.sum(z1, dim=1))
        r2 = self.model.decoder(torch.sum(z2, dim=1))

        x_labels = ('Image 1', 'Reconstructed 1', 'Exchanged', 'Reconstructed 2', 'Image 2')

        for i in range(6):
            sample = random.sample(list(range(6)), k=random.randrange(1, 5))

            z_exch = self.exchange_feature(z1, z2, sample)
            z_exch = torch.sum(z_exch, dim=1)

            reconstruction = self.model.decoder(z_exch)
            for j, img in enumerate([img1[0], r1[0], reconstruction[0], r2[0], img2[0]]):
                ax[i, j].imshow(img.detach().cpu().numpy().transpose(1, 2, 0), cmap='gray')
                if j == 0:
                    ax[i, j].set_ylabel(make_y_label_name(sample))
                if i == 5:
                    ax[i, j].set_xlabel(x_labels[j])
                if j != 0 and i != 5:
                    ax[i, j].set_axis_off()
        fig.tight_layout()
        wandb.log({"Swap random features": plt})

    def decode_features(self):
        fig, ax = plt.subplots(1, 6, figsize=(10, 8))

        batch = next(iter(self.loader))
        img, _img, _exchange_labels = batch

        z = self.model.encode_features_latent(img.to(self.device))

        ax[0].imshow(img[0].detach().cpu().numpy().transpose(1, 2, 0), cmap='gray')
        ax[0].set_axis_off()

        for i in range(6):
            feature = z[:, i]
            r = self.model.decoder(feature)[0]

            ax[i + 1].imshow(r.detach().cpu().numpy().transpose(1, 2, 0), cmap='gray')
            ax[i + 1].set_axis_off()
        fig.tight_layout()
        wandb.log({"Decode image from feature": plt})


if __name__ == '__main__':
    experiment = Experiment(args.checkpoint_path)
    for i in range(args.one):
        experiment.swap_one_feature()
    for i in range(args.random):
        experiment.swap_random_features()
    for i in range(args.several):
        experiment.swap_several_features()
    for i in range(args.decode):
        experiment.decode_features()
