import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import AdamW
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchmetrics.functional import dice
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

seg_models = {
    "Unet": smp.Unet,
    "Unet++": smp.UnetPlusPlus,
    "MAnet": smp.MAnet,
    "Linknet": smp.Linknet,
    "FPN": smp.FPN,
    "PSPNet": smp.PSPNet,
    "PAN": smp.PAN,
    "DeepLabV2": smp.DeepLabV3,
    "DeepLabV2+": smp.DeepLabV3Plus,
}

class LightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = seg_models[config["seg_model"]](
            encoder_name=config["encoder_name"],
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )
        self.loss_module = smp.losses.DiceLoss(mode="binary", smooth=config["loss_smooth"])
        self.val_step_outputs = []
        self.val_step_labels = []
        self.train_step_outputs = []
        self.train_step_labels = []
    
    def forward(self, batch):
        preds = self.model(batch)
        return preds
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.config["optimizer_params"])
        return {"optimizer": optimizer}
    
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        if self.config["image_size"] != 512:
            preds = torch.nn.functional.interpolate(preds, size=512, mode='bilinear')
        loss = self.loss_module(preds, labels)
        self.log("Train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_step_outputs.append(preds)
        self.train_step_labels.append(labels)
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        if self.config["image_size"] != 512:
            preds = torch.nn.functional.interpolate(preds, size=512, mode='bilinear')
        loss = self.loss_module(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append(preds)
        self.val_step_labels.append(labels)
    
    def __create_preds_labels(self, outputs, labels):
        if len(outputs) == 0 or len(labels) == 0: 
            return None, None
        all_preds = torch.cat(outputs).cpu()
        all_labels = torch.cat(labels).cpu()
        all_preds = all_preds.to(torch.float32) 
        all_labels = all_labels.to(torch.float32) 
        all_preds = torch.sigmoid(all_preds)
        return all_preds, all_labels
    
    def __clear_lists(self) -> None:
        self.val_step_outputs.clear()
        self.val_step_labels.clear()
        self.train_step_outputs.clear()
        self.train_step_labels.clear()

    def __calculate_dice(self, preds, labels): 
        if preds is None:
            return -1
        dice_coef = dice(preds, labels.long())
        return dice_coef
    
    def on_validation_epoch_end(self):
        train_preds, train_labels = self.__create_preds_labels(self.train_step_outputs, self.train_step_labels)
        val_preds, val_labels = self.__create_preds_labels(self.val_step_outputs, self.val_step_labels)
        self.__clear_lists()
        val_dice = self.__calculate_dice(val_preds, val_labels)
        train_dice = self.__calculate_dice(train_preds, train_labels)
        self.log("val_dice", val_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_dice", train_dice, on_step=False, on_epoch=True, prog_bar=True)
        if self.trainer.global_rank == 0:
            print(f"\nEpoch: {self.current_epoch}", flush=True)




