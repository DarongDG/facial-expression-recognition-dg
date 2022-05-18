from pytorch_lightning import LightningModule
import torch as t
import torch.nn.functional as F
from argparse import Namespace

import matplotlib.pyplot as plt
import ipdb
import os
import torchmetrics
from torchmetrics.classification import accuracy
#

from dl.base_lightning_modules.plotter import plot_train_loss


class BaseClassificationModel(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.save_hyperparameters()
        self.generator = t.nn.Sequential()
        self.loss = t.nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.num_classes = 7
        self.iteration = 0
        self.train_loss_list = []
        self.val_loss_list = []

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.generator(z)
        return out

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        self.iteration += 1
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        if self.iteration % 50 == 0:
            self.train_loss_list.append((self.iteration,loss.item()))
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        acc = self.val_accuracy.compute()
        self.log("accuracy", acc, prog_bar=True)
        self.log("val_loss", 1 - acc, prog_bar=True)
        avg_loss = t.stack([x["val_loss_ce"] for x in outputs]).mean()
        self.log("val_loss_ce", avg_loss, prog_bar=True)
        self.val_loss_list.append((self.iteration, avg_loss.item()))
        self.val_accuracy.reset()
        t.save(
            self.state_dict(), os.path.join(self.params.save_path, "checkpoint.ckpt"),
        )
        plot_train_loss(self.train_loss_list, self.val_loss_list,
                        save_path=self.params.save_path)

        return {"val_loss": acc}

    def training_epoch_end(self, outputs):
        avg_loss = t.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        if batch_idx == 0:
            pass
        pred_y = self(x)
        loss = self.loss(pred_y, y).to(self.device)
        self.val_accuracy.update(pred_y, y)
        return {"val_loss_ce": loss}

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        if batch_idx == 0:
            pass
        pred_y = self(x)
        self.test_accuracy.update(pred_y, y)
        return 

    def test_epoch_end(self, outputs):
        accuracy = self.test_accuracy.compute()
        self.test_accuracy.reset()
        test_metrics = {
            "accuracy": accuracy,
        }
        test_metrics = {k: v for k, v in test_metrics.items()}
        self.log("lol", test_metrics, prog_bar=True)

    def configure_optimizers(self):
        lr = self.params.lr
        b1 = self.params.b1
        b2 = self.params.b2

        optimizer = t.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2)
        )
        
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=self.params.reduce_lr_on_plateau_patience,
            min_lr=1e-6,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
