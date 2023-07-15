import warnings
import sys

sys.path.append("../data")
warnings.filterwarnings("ignore")

import gc
import os
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
from utils import load_yaml, create_df, create_folds
import pandas as pd
from data.train_pipeline import HubMAP_Dataset
from model import LighningModule

CONFIG_PATH = "D:\Machine_Learning\hubmap-hacking-the-human-vasculature\src\models\config\unet.yaml"

torch.set_float32_matmul_precision("medium")

config = load_yaml()

pl.seed_everything(config["seed"])

gc.enable()

df = create_df(config=config, train_folder="train", polygons_json="polygons.jsonl")

df = create_folds(config=config, df=df)

for fold in config["train_folds"]:
    print(f"\n###### Fold {fold}")
    trn_df = df[df.kfold != fold].reset_index(drop=True)
    vld_df = df[df.kfold == fold].reset_index(drop=True)

    dataset_train = HubMAP_Dataset(trn_df, config["model"]["image_size"], train=True)
    dataset_validation = HubMAP_Dataset(vld_df, config["model"]["image_size"], train=False)
    
    
#     image, mask = dataset_train[0]
#     fig, axs = plt.subplots(1, 2, figsize=(16, 8))
#     #axs[0].imshow(image)
#     axs[1].imshow(mask)
#     print(f"Image: {image.shape}, Mask: {mask.shape}")
    
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config["train_bs"],
        shuffle=True,
        num_workers=config["workers"],
    )
    data_loader_validation = DataLoader(
        dataset_validation,
        batch_size=config["valid_bs"],
        shuffle=False,
        num_workers=config["workers"],
    )

    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        monitor="val_dice",
        dirpath=config["output_dir"],
        mode="max",
        filename=f"model-f{fold}-{{val_dice:.4f}}",
        save_top_k=1,
        verbose=1,
    )

    progress_bar_callback = TQDMProgressBar(
        refresh_rate=config["progress_bar_refresh_rate"]
    )

    early_stop_callback = EarlyStopping(**config["early_stop"])


    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback, progress_bar_callback],
        logger=CSVLogger(save_dir=f'logs_f{fold}/'),
        **config["trainer"],
    )

    model = LightningModule(config["model"])

    trainer.fit(model, data_loader_train, data_loader_validation)

    del (
        dataset_train,
        dataset_validation,
        data_loader_train,
        data_loader_validation,
        model,
        trainer,
        checkpoint_callback,
        progress_bar_callback,
        early_stop_callback,
    )
    torch.cuda.empty_cache()
    gc.collect()