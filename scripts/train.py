import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import torchvision.transforms as T
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from torchinfo import summary 

from scripts.dataset import ImageNet100Dataset
from models.standard_resnet import StandardResNet
from PIL import Image

class ImageNet100Classifier(pl.LightningModule):
    def __init__(self, lr=0.1, max_epochs=120, warmup_epochs=10):
        super().__init__()
        self.save_hyperparameters()
        self.model = StandardResNet(num_classes=100, num_extra_resblocks=3)
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses, self.train_accs = [], []
        self.val_losses, self.val_accs = [], []

        self.last_train_loss = None
        self.last_train_acc = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.train_losses.append(loss.detach())
        self.train_accs.append(acc.detach())
        return loss

    def on_train_epoch_end(self):
        self.last_train_loss = torch.stack(self.train_losses).mean()
        self.last_train_acc = torch.stack(self.train_accs).mean()
        self.train_losses.clear()
        self.train_accs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.val_losses.append(loss.detach())
        self.val_accs.append(acc.detach())
        return loss

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.val_losses).mean()
        avg_val_acc = torch.stack(self.val_accs).mean()

        log_dict = {
            "train/loss": self.last_train_loss if self.last_train_loss is not None else torch.tensor(0.0),
            "train/acc": self.last_train_acc if self.last_train_acc is not None else torch.tensor(0.0),
            "val/loss": avg_val_loss,
            "val/acc": avg_val_acc
        }

        self.log_dict(log_dict, prog_bar=True, logger=True)

        self.val_losses.clear()
        self.val_accs.clear()

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=1e-4
        )

        warmup = LinearLR(optimizer, start_factor=1.0 / self.hparams.warmup_epochs,
                          total_iters=self.hparams.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer,
                                   T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
                                   eta_min=0.0)

        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                 milestones=[self.hparams.warmup_epochs])
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}



if __name__ == "__main__":
    root_dir = "/root/workspace/imagenet100"
    train_folders = ["train.X1", "train.X2", "train.X3", "train.X4"]
    val_folder = ["val.X"]
    batch_size = 256
    accumulate_grad_batches = 1
    max_epochs = 120

    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_set = ImageNet100Dataset(root_dir, train_folders, train_transform)
    val_set = ImageNet100Dataset(root_dir, val_folder, val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    csv_logger = CSVLogger(save_dir="logs", name="imagenet100")

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_cb = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="StandardResNet_0"
    )
    model = ImageNet100Classifier(lr=0.1, max_epochs=max_epochs, warmup_epochs=10)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        logger=csv_logger,
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[lr_monitor, checkpoint_cb, RichProgressBar()],
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, val_loader)
