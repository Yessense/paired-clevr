from argparse import ArgumentParser
from typing import Tuple

import pytorch_lightning as pl
import torch.optim
import vsa  # type: ignore
import wandb
from vsa import *

from src.model.utils import iou_pytorch

torch.set_printoptions(sci_mode=False)

from src.model.decoder import Decoder  # type: ignore
from src.model.encoder import Encoder  # type: ignore

torch.manual_seed(42)

EPSILON = 1e-6


class ClevrVAE(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("MnistSceneEncoder")

        # dataset options
        parser.add_argument("--n_features", type=int, default=6)
        parser.add_argument("--image_size", type=Tuple[int, int, int], default=(3, 240, 320))  # type: ignore

        # model options
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--kld_coef", type=float, default=0.01)
        parser.add_argument("--loss_mode", type=str, default='mean')

        # debug options
        parser.add_argument("--save_batch_errors", type=bool, default=False)

        # experiment options
        parser.add_argument("--latent_dim", type=int, default=1024)
        parser.add_argument("--hd_norm", type=bool, default=False)

        return parent_parser

    def __init__(self,
                 n_features: int = 6,
                 image_size: Tuple[int, int, int] = (1, 64, 64),
                 lr: float = 0.001,
                 kld_coef: float = 0.01,
                 loss_mode: str = 'mean',
                 save_batch_errors: bool = False,
                 latent_dim: int = 1024,
                 hd_norm: bool = False,
                 **kwargs):
        super().__init__()
        # debug
        self.save_batch_errors = save_batch_errors
        self.previous_loss = 0.

        # Experiment options
        self.latent_dim = latent_dim
        self.n_features = n_features
        self.hd_norm = hd_norm

        # model parameters
        self.lr = lr
        self.kld_coef = kld_coef
        self.loss_mode = loss_mode
        self.img_dim = image_size
        self.encoder = Encoder(latent_dim=latent_dim, image_size=image_size, n_features=n_features)
        self.decoder = Decoder(latent_dim=latent_dim, image_size=image_size, n_features=n_features)

        # hd placeholders
        self.hd_placeholders = torch.randn(1, self.n_features, self.latent_dim)
        if self.hd_norm:
            self.hd_placeholders /= torch.linalg.norm(self.hd_placeholders, dim=2).unsqueeze(-1)

        self.save_hyperparameters()

    def forward(self, x):
        mu, log_var = self.encoder(x)
        return mu

    def get_latent(self, x):
        mu, log_var = self.encoder(x.to(self.device))
        z = mu.view(-1, self.n_features, self.latent_dim)
        z = torch.sum(z, dim=1)
        return z

    def reconstruct(self, z):
        return self.decoder(z.to(self.device))

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            # Reconstruction mode
            return mu

    def encode_features_inference(self, img):
        z, log_var = self.encoder(img)
        z = z.view(-1, self.n_features, self.latent_dim)
        mask = self.hd_placeholders.expand(z.size()).to(self.device)
        z = z * mask
        return z

    def encode_features(self, img):
        mu, log_var = self.encoder(img)
        z = self.reparameterize(mu, log_var)
        z = z.view(-1, self.n_features, self.latent_dim)
        mask = self.hd_placeholders.expand(z.size()).to(self.device)
        z = z * mask

        return mu, log_var, z

    def validation_step(self, batch, idx):
        # img -> (64, 64)
        # exchange
        img1, img2, exchange_labels = batch

        mu1, log_var1, feat_1 = self.encode_features(img1)
        mu2, log_var2, feat_2 = self.encode_features(img2)

        exchange_labels = exchange_labels.expand(feat_1.size())

        # z1 Восстанавливает 1 изображение
        z1 = torch.where(exchange_labels, feat_1, feat_2)
        z1 = torch.sum(z1, dim=1)
        r1 = self.decoder(z1)

        # z2 Восстанавливает 2 изображение изображение
        z2 = torch.where(exchange_labels, feat_2, feat_1)
        z2 = torch.sum(z2, dim=1)
        r2 = self.decoder(z2)

        # calculate loss
        assert torch.all(r1 >= 0) and torch.all(r1 <= 1)
        assert torch.all(r2 >= 0) and torch.all(r2 <= 1)

        iou1 = iou_pytorch(r1, img1)
        iou2 = iou_pytorch(r2, img2)
        iou = (iou1 + iou2) / 2

        mu = mu1 + mu2 / 2
        log_var = log_var1 + log_var2 / 2
        loss = self.loss_f(r1, r2, img1, img2, mu, log_var)

        # # log training process
        self.log("Val iou", iou, prog_bar=False)
        self.log("Val iou image 1", iou1, prog_bar=False)
        self.log("Val iou image 2", iou2, prog_bar=False)
        self.log("Validation Loss", loss[0], prog_bar=False)

        # 67 batches with 16
        if idx % 10 == 0:
            self.logger.experiment.log({
                "reconstruct/validation": [
                    wandb.Image(img1[0], caption='Val Image 1'),
                    wandb.Image(img2[0], caption='Val Image 2'),
                    wandb.Image(r1[0], caption='Val Recon 1'),
                    wandb.Image(r2[0], caption='Val Recon 2'),
                ]})

    def training_step(self, batch):
        """Exchanges, loss, logs"""

        # ----------------------------------------
        # Exchange
        # ----------------------------------------

        img1, img2, exchange_labels = batch

        # Encode features
        mu1, log_var1, feat_1 = self.encode_features(img1)
        mu2, log_var2, feat_2 = self.encode_features(img2)

        exchange_labels = exchange_labels.expand(feat_1.size())

        # z1 Восстанавливает 1 изображение
        z1 = torch.where(exchange_labels, feat_1, feat_2)
        z1 = torch.sum(z1, dim=1)
        r1 = self.decoder(z1)

        # z2 Восстанавливает 2 изображение
        z2 = torch.where(exchange_labels, feat_2, feat_1)
        z2 = torch.sum(z2, dim=1)
        r2 = self.decoder(z2)

        # Check numerical stability
        assert torch.all(r1 >= 0) and torch.all(r1 <= 1)
        assert torch.all(r2 >= 0) and torch.all(r2 <= 1)

        # ----------------------------------------
        # Loss
        # ----------------------------------------

        mu = mu1 + mu2 / 2
        log_var = log_var1 + log_var2 / 2
        loss = self.loss_f(r1, r2, img1, img2, mu, log_var)

        # ----------------------------------------
        # Debug
        # ----------------------------------------

        # check for error by comparing previous loss with current
        if self.save_batch_errors:
            if self.current_epoch > 7 and (self.previous_loss / loss[0].item() > 2):
                torch.save(batch, f'check_batch_{self.global_step}_{self.previous_loss / loss[0].item():.1f}.pt')
            self.previous_loss = loss[0].item()

        # ----------------------------------------
        # Metrics
        # ----------------------------------------

        iou1 = iou_pytorch(r1, img1)
        iou2 = iou_pytorch(r2, img2)
        iou = (iou1 + iou2) / 2

        # ----------------------------------------
        # Logs
        # ----------------------------------------

        self.log("Total loss", loss[0], prog_bar=False)
        self.log("Reconstruct image 1", loss[1], prog_bar=False)
        self.log("Reconstruct image 2", loss[2], prog_bar=False)
        self.log("kld", loss[3], prog_bar=False)
        self.log("iou", iou, prog_bar=True)
        self.log("iou image 1", iou1, prog_bar=False)
        self.log("iou image 2", iou2, prog_bar=False)

        if self.global_step % 500 == 0:
            self.logger.experiment.log({
                "reconstruct/train": [
                    wandb.Image(img1[0], caption='Image 1'),
                    wandb.Image(img2[0], caption='Image 2'),
                    wandb.Image(r1[0], caption='Recon 1'),
                    wandb.Image(r2[0], caption='Recon 2'),
                ]})
        return loss[0]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def loss_f(self, r1, r2, scene1, scene2, mu, log_var):
        if self.loss_mode == 'mean':
            batch_size = r1.shape[0]
            loss = torch.nn.BCELoss(reduction='mean')

            l1 = loss(r1, scene1)
            l2 = loss(r2, scene2)

            # KLD loss
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kld /= batch_size
        elif self.loss_mode == 'sum':
            loss = torch.nn.BCELoss(reduction='sum')

            l1 = loss(r1, scene1)
            l2 = loss(r2, scene2)

            # KLD loss
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        else:
            raise KeyError

        return l1 + l2 + self.kld_coef * kld, l1, l2, self.kld_coef * kld
