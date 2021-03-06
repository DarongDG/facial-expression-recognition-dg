import pandas as pd
from pytorch_lightning import LightningModule
import torch as t
import torch.nn.functional as F
from argparse import Namespace

import matplotlib.pyplot as plt
import ipdb
import os
import torchmetrics
from torchmetrics.classification import accuracy
import numpy
import seaborn as sns

#

from dl.base_lightning_modules.plotter import plot_train_loss, plot_acc


class MetricTracker:
    def __init__(self, num_classes=7, key="train", *args, **kwargs):

        self.key = key
        self.num_classes = num_classes

        self.metrics = {
            "F1-Score": torchmetrics.F1Score(num_classes=num_classes, average="macro"),
            "Precision": torchmetrics.Precision(
                num_classes=num_classes, average="macro"
            ),
            "Recall": torchmetrics.Recall(num_classes=num_classes, average="macro"),
            "Confusion Matrix": torchmetrics.ConfusionMatrix(
                num_classes=num_classes, normalize="true"
            ),
            "Accuracy": torchmetrics.Accuracy(),
        }

    def get_metrics(self):
        return {
            metric_name: metric.compute()
            for metric_name, metric in self.metrics.items()
        }

    def update(self, output, target):

        # set both devices to same
        output = output
        target = target

        # ipdb.set_trace()
        for metric in self.metrics.values():
            metric.update(output.detach().cpu(), target.detach().cpu())

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def log(self, logger, epoch):
        for metric_name, metric in self.metrics.items():
            if metric_name == "Confusion Matrix":
                self.save_confusion_matrix(
                    logger, metric.compute().detach().numpy(), epoch
                )

            else:
                logger.experiment.add_scalar(
                    f"{self.key}_{metric_name}/epoch", metric.compute().detach(), epoch
                )
            metric.reset()

    def save_confusion_matrix(self, logger, confusion_matrix, epoch):

        df_cm = pd.DataFrame(
            confusion_matrix,
            index=range(self.num_classes),
            columns=range(self.num_classes),
        )
        plt.figure(figsize=(7, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
        plt.close(fig_)
        logger.experiment.add_figure(f"{self.key} Confusion", fig_, epoch)


class BaseClassificationModel(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.save_hyperparameters()
        self.num_classes = 7

        self.generator = t.nn.Sequential()
        self.loss = t.nn.CrossEntropyLoss()

        self.val_accuracy = torchmetrics.Accuracy()
        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        self.test_metrics = MetricTracker(self.num_classes, key="Test")
        self.train_metrics = MetricTracker(self.num_classes, key="Train")
        self.val_metrics = MetricTracker(self.num_classes, key="Val")

        self.iteration = 0
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []
        self.rand_img = None

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.generator(z)
        return out

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        self.iteration += 1
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.train_metrics.update(y_pred, y)
        if self.iteration % 50 == 0:
            self.train_loss_list.append((self.iteration, loss.item()))
        self.logger.experiment.add_scalar("loss/iteration", loss, self.iteration)
        self.train_accuracy.update(y_pred, y)

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        acc = self.val_accuracy.compute()
        self.log("val_accuracy", acc, prog_bar=True)
        self.val_acc_list.append((self.iteration, acc.item()))
        avg_loss = t.stack([x["val_loss_ce"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        self.val_loss_list.append((self.iteration, avg_loss.item()))
        self.val_accuracy.reset()
        t.save(
            self.state_dict(),
            os.path.join(self.params.save_path, "checkpoint.ckpt"),
        )
        plot_acc(self.val_acc_list, save_path=self.params.save_path)
        plot_train_loss(
            self.train_loss_list, self.val_loss_list, save_path=self.params.save_path
        )
        self.logger.experiment.add_scalar(
            "val_loss/epoch", avg_loss, self.current_epoch
        )

        self.val_metrics.log(self.logger, self.current_epoch)

        return {"val_loss": avg_loss}

    def training_epoch_end(self, outputs):
        train_acc = self.train_accuracy.compute()
        self.log("train_accuracy", train_acc, prog_bar=True)
        self.train_acc_list.append((self.iteration, train_acc.item()))
        self.train_accuracy.reset()
        avg_loss = t.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar(
            "train_loss/epoch", avg_loss, self.current_epoch
        )

        if self.current_epoch == 0:
            sample = t.zeros(1, 1, 48, 48)
            sample = sample.to(self.device)
            self.logger.experiment.add_graph(self.generator, sample)

        self.train_metrics.log(self.logger, self.current_epoch)

    def validation_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        if batch_idx == 0:
            pass
        pred_y = self(x)
        loss = self.loss(pred_y, y).to(self.device)
        self.val_accuracy.update(pred_y, y)
        self.logger.experiment.add_scalar("loss/iteration", loss, self.iteration)

        self.val_metrics.update(pred_y, y)

        return {"val_loss_ce": loss}

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        if batch_idx == 0:
            pass
        pred_y = self(x)
        # self.test_accuracy.update(pred_y, y)
        loss = self.loss(pred_y, y).to(self.device)
        self.logger.experiment.add_scalar("loss/iteration", loss, self.iteration)
        if self.rand_img is None:
            self.rand_img = x
        self.test_metrics.update(pred_y, y)

        return

    def test_epoch_end(self, outputs):
        # accuracy = self.test_accuracy.compute()
        # self.test_accuracy.reset()
        # # self.log("test_accuracy", accuracy, prog_bar=True)
        # self.test_accuracy.reset()
        # self.logger.experiment.add_scalar(
        #     "test_accuracy/epoch", accuracy, self.current_epoch
        # )
        self.showActivations(self.rand_img)
        metrics = self.test_metrics.get_metrics()
        self.test_metrics.log(self.logger, self.current_epoch)
        for key, value in metrics.items():
            self.log(key, value, prog_bar=True)

    def configure_optimizers(self):
        lr = self.params.lr
        b1 = self.params.b1
        b2 = self.params.b2

        optimizer = t.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))

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

    """
    necessary for activation visualization
    """

    def makegrid(self, output, numrows):
        outer = t.Tensor.cpu(output).detach()
        plt.figure(figsize=(20, 5))
        b = numpy.array([]).reshape(0, outer.shape[2])
        c = numpy.array([]).reshape(numrows * outer.shape[2], 0)
        i = 0
        j = 0
        while i < outer.shape[1]:
            img = outer[0][i]
            b = numpy.concatenate((img, b), axis=0)
            j += 1
            if j == numrows:
                c = numpy.concatenate((c, b), axis=1)
                b = numpy.array([]).reshape(0, outer.shape[2])
                j = 0

            i += 1
        return c

    def showActivations(self, x):
        # logging reference image
        rand_first = t.randint(0, x.shape[0], (1,))[0]
        detached = x[rand_first, ...].cpu().detach() * 0.5 + 0.5
        x = x[rand_first, ...].unsqueeze(0)
        self.logger.experiment.add_image("input", detached, self.current_epoch)

        # logging layer 1 activations
        out = self.generator.conv1(x)
        c = self.makegrid(out, 4)
        self.logger.experiment.add_image(
            "layer 1", c, self.current_epoch, dataformats="HW"
        )

        # visualize kernels
        # ipdb.set_trace()
        kernel = self.generator.conv1.conv.weight.cpu()
        # ipdb.set_trace()
        # kernel = self.makegrid(kernel, 1)
        kernel = kernel[:64, ...]
        self.logger.experiment.add_image(
            "kernel 1",
            kernel,
            self.current_epoch,
            dataformats="NCHW",
        )

        # logging layer 1 activations
        out = self.generator.conv2(out)
        c = self.makegrid(out, 8)
        self.logger.experiment.add_image(
            "layer 2", c, self.current_epoch, dataformats="HW"
        )
        kernel = self.generator.conv2.conv.weight.cpu()
        kernel = self.makegrid(kernel, 8)
        self.logger.experiment.add_image(
            "kernel 2", kernel, self.current_epoch, dataformats="HW"
        )
        # # logging layer 1 activations
        out = self.generator.conv3(out)
        c = self.makegrid(out, 8)
        self.logger.experiment.add_image(
            "layer 3", c, self.current_epoch, dataformats="HW"
        )
        kernel = self.generator.conv3.conv.weight.cpu()
        kernel = self.makegrid(kernel, 8)
        self.logger.experiment.add_image(
            "kernel 3", kernel, self.current_epoch, dataformats="HW"
        )

        # conv3
        out = self.generator.conv4(out)
        c = self.makegrid(out, 8)
        self.logger.experiment.add_image(
            "layer 4", c, self.current_epoch, dataformats="HW"
        )
        kernel = self.generator.conv4.conv.weight.cpu()

        kernel = self.makegrid(kernel, 8)
        self.logger.experiment.add_image(
            "kernel 4", kernel, self.current_epoch, dataformats="HW"
        )
