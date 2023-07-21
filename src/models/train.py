import warnings
import sys
import wandb

sys.path.append("../../src")
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
from model import LightningModule
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":
    BASE_DIR = "D:/Machine_Learning/hubmap-hacking-the-human-vasculature/"
    CONFIG_PATH = os.path.join(BASE_DIR, "src/models/config/unet-0.yaml")

    print(CONFIG_PATH)
    torch.set_float32_matmul_precision("medium")

    config = load_yaml(CONFIG_PATH)

    pl.seed_everything(config["seed"])

    gc.enable()

    df = create_df(config=config, train_folder="train", polygons_json="polygons.jsonl")

    df = create_folds(config=config, df=df)

    df.to_csv(os.path.join(BASE_DIR, "data/train_df.csv"), index=False)

    for fold in config["train_folds"]:
        print(f"\n###### Fold {fold}")
        trn_df = df[df.kfold != fold].reset_index(drop=True)
        vld_df = df[df.kfold == fold].reset_index(drop=True)

        dataset_train = HubMAP_Dataset(
            trn_df, config["model"]["image_size"], train=True
        )
        dataset_validation = HubMAP_Dataset(
            vld_df, config["model"]["image_size"], train=False
        )

        image, mask = dataset_train[0]
        print(f"Image Shape: {image.shape}, Mask Shape: {mask.shape}")

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

        wandb_logger = WandbLogger(project="my-awesome-project")

        wandb_logger.experiment.config["batch_size"] = config["train_bs"]

        trainer = pl.Trainer(
            callbacks=[
                checkpoint_callback,
                early_stop_callback,
                progress_bar_callback,
            ],
            logger=[CSVLogger(save_dir=f"logs_f{fold}/"), wandb_logger],
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

        wandb.finish()

        break
