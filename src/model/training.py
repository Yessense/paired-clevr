import os.path
import sys

import numpy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append("..")
from pytorch_lightning.loggers import WandbLogger

from src.dataset.dataset import PairedClevr
from src.model.scene_vae import ClevrVAE
import pytorch_lightning as pl
from argparse import ArgumentParser

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

DEFAULT_SEED = 42
DEFAULT_DATASET_PATH = './dataset/data/'
DEFAULT_IMAGES_DIR_PATH = os.path.join(DEFAULT_DATASET_PATH, 'images')
DEFAULT_SCENES_DIR_PATH = os.path.join(DEFAULT_DATASET_PATH, 'scenes')
DEFAULT_LOGGER_DIR_PATH = './'
# ------------------------------------------------------------
# Parse args
# ------------------------------------------------------------

parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')

# logger parameters
program_parser.add_argument("--logger_dir", type=str, default=DEFAULT_LOGGER_DIR_PATH)
program_parser.add_argument("--log_model", default=True)

# dataset parameters
program_parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
program_parser.add_argument("--images_path", type=str, default=DEFAULT_IMAGES_DIR_PATH)
program_parser.add_argument("--scenes_path", type=str, default=DEFAULT_SCENES_DIR_PATH)

# Experiment parameters
program_parser.add_argument("--batch_size", type=int, default=2)
program_parser.add_argument("--from_checkpoint", type=str, default='')
program_parser.add_argument("--grad_clip", type=float, default=0.0)
program_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

# Add model specific args
parser = ClevrVAE.add_model_specific_args(parent_parser=parser)

# Add all the available trainer options to argparse#
parser = pl.Trainer.add_argparse_args(parser)

# Parse input
args = parser.parse_args()

# ------------------------------------------------------------
# Logger
# ------------------------------------------------------------

wandb_logger = WandbLogger(project='paired-clevr', log_model=args.log_model, save_dir=args.logger_dir)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

train, test = train_test_split(numpy.arange(10000), test_size=0.1, random_state=args.seed)
train_dataset = PairedClevr(scenes_dir=args.scenes_path, img_dir=args.images_path, indices=train)
test_dataset = PairedClevr(scenes_dir=args.scenes_path, img_dir=args.images_path, indices=test)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

# model
dict_args = vars(args)
autoencoder = ClevrVAE(**dict_args)

# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

monitor = 'Validation Loss'

# early stop
patience = 5
early_stop_callback = EarlyStopping(monitor=monitor, patience=patience)

# checkpoints
save_top_k = 2
checkpoint_callback = ModelCheckpoint(monitor=monitor, save_top_k=save_top_k)

# stohastic weight averaging
swa_lrs = 1e-2
swa = StochasticWeightAveraging(swa_lrs=swa_lrs)

callbacks = [
    checkpoint_callback,
    # swa,
    # early_stop_callback,
]

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

# trainer parameters
profiler = 'simple'  # 'simple'/'advanced'/None
gpus = [args.gpus]
log_every_n_steps = None

# trainer
trainer = pl.Trainer(gpus=gpus,
                     max_epochs=args.max_epochs,
                     profiler=profiler,
                     callbacks=callbacks,
                     logger=wandb_logger,
                     gradient_clip_val=args.grad_clip)

if not len(args.from_checkpoint):
    args.from_checkpoint = None
trainer.fit(autoencoder, train_dataloaders=train_loader, val_dataloaders=test_loader, ckpt_path=args.from_checkpoint)
