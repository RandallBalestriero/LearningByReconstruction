from pathlib import Path
from omega import utils
import omega


import torch
import torchmetrics
import utils
import lightning.pytorch as pl

torch.set_float32_matmul_precision("medium")
import matplotlib.pyplot as plt
import aidatasets
from torch.utils.data import DataLoader
import omega.pl
from torchvision.transforms import v2

plt.style.use("seaborn-v0_8-poster")


class MyDataset(pl.LightningDataModule):
    def __init__(self, path, batch_size, dataset, augment):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        if "CIFAR" in dataset:
            self.train_transform = v2.Compose(
                [v2.RandomHorizontalFlip(p=0.5), v2.ToDtype(torch.float32, scale=True)]
            )
            self.val_transform = v2.ToDtype(torch.float32, scale=True)
            self.shape = (3, 32, 32)
        elif "Imagenet" in dataset:
            if augment:
                self.train_transform = v2.Compose(
                    [
                        v2.PILToTensor(),
                        v2.RandomResizedCrop(size=224, scale=(0.5, 1)),
                        v2.RandomHorizontalFlip(p=0.5),
                        v2.RandomGrayscale(p=0.2),
                        v2.ToDtype(torch.float32, scale=True),
                    ]
                )
            else:
                self.train_transform = v2.Compose(
                    [
                        v2.PILToTensor(),
                        v2.Resize(256),
                        v2.CenterCrop(224),
                        v2.ToDtype(torch.float32, scale=True),
                    ]
                )
            self.val_transform = v2.Compose(
                [
                    v2.PILToTensor(),
                    v2.Resize(256),
                    v2.CenterCrop(224),
                    v2.ToDtype(torch.float32, scale=True),
                ]
            )
            self.shape = (3, 224, 224)
        self.dataset = dataset

    def setup(self, stage: str = None):
        if hasattr(self, "train"):
            return
        dataset = (
            aidatasets.images.__dict__[self.dataset](path=self.path).download().load()
        )

        self.train = aidatasets.utils.TensorDataset(
            dataset["train_X"],
            dataset["train_y"],
            transform=self.train_transform,
        )
        self.val = aidatasets.utils.TensorDataset(
            dataset["test_X"],
            dataset["test_y"],
            transform=self.val_transform,
        )
        print(len(self.train), len(self.val))

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=20,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)


class MyLightningModule(omega.pl.Module):
    def create_modules(self):
        if self.arch == "DCAE":
            self.model = utils.DCAE(
                self.latent,
                encoder_depth=self.encoder_depth,
                decoder_depth=self.decoder_depth,
            )
        elif self.arch == "MLPAE":
            self.model = utils.MLPAE(
                self.input_shape,
                self.latent,
                encoder_depth=self.encoder_depth,
                decoder_depth=self.decoder_depth,
            )
        elif self.arch == "R34":
            self.model = utils.ResNetAutoEncoder(
                [3, 4, 6, 3], False, (512, 7, 7), self.latent
            )
            self.probe = torch.nn.Sequential(
                # torch.nn.AdaptiveAvgPool2d(1),
                # torch.nn.Flatten(),
                torch.nn.Linear(self.latent, 10),
            )
            self.nl_probe = torch.nn.Sequential(
                # torch.nn.AdaptiveAvgPool2d(1),
                # torch.nn.Flatten(),
                torch.nn.Linear(self.latent, 1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 10),
            )
            if self.guidance == 0:
                self.model.decoder.requires_grad_(False)

    def create_metrics(self):
        self.train_error = torchmetrics.MeanSquaredError()
        self.val_error = torchmetrics.MeanSquaredError()
        self.acc = torchmetrics.classification.MulticlassAccuracy(10)
        self.nl_acc = torchmetrics.classification.MulticlassAccuracy(10)
        self.val_acc = torchmetrics.classification.MulticlassAccuracy(10)
        self.nl_val_acc = torchmetrics.classification.MulticlassAccuracy(10)

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch):
        return self.val_acc(self.probe(self.model.encoder(batch[0])), batch[1])

    def compute_loss(self, batch, batch_idx):
        x, y = batch
        h = self.model.encoder(x)
        preds = self.probe(h)
        nl_preds = self.nl_probe(h)
        sup_loss = torch.nn.functional.cross_entropy(preds, y)
        nl_sup_loss = torch.nn.functional.cross_entropy(nl_preds, y)
        if self.guidance == 1.0:
            xrec = self.model.decoder(h)
            rec_loss = torch.nn.functional.mse_loss(xrec, x)
        else:
            rec_loss = 0
        self.log("train_rec_loss", rec_loss)
        self.acc(preds, y)
        self.nl_acc(nl_preds, y)
        self.log("train_acc", self.acc, on_step=False, on_epoch=True)
        self.log("nl_train_acc", self.nl_acc, on_step=False, on_epoch=True)
        return rec_loss + sup_loss + nl_sup_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        h = self.model.encoder(x)
        xrec = self.model.decoder(h)
        preds = self.probe(h)
        nl_preds = self.nl_probe(h)
        self.val_error(xrec, x)
        self.val_acc(preds, y)
        self.nl_val_acc(nl_preds, y)
        self.log("valid_loss", self.val_error, on_step=False, on_epoch=True)
        self.log("valid_acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("nl_valid_acc", self.nl_val_acc, on_step=False, on_epoch=True)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path
    import omega
    import aidatasets

    parser = ArgumentParser()
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--arch", type=lambda x: x.split(","), default=["MLPAE"])
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--latent", type=int, default=128)
    parser.add_argument("--guidance", type=float, default=0)
    parser.add_argument("--dataset", type=lambda x: x.split(","), default=["CIFAR10"])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--corruption", action="store_true")
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()
    guidance = args.guidance

    for dataset in args.dataset:
        for arch in args.arch:
            for depth in [3]:
                score = {}
                path = (
                    args.path
                    / "guidance"
                    / f"ae_{dataset.upper()}_{arch}_{args.augment}__{args.latent}_{guidance}".replace(
                        ".", "-"
                    )
                )
                datamodule = MyDataset(
                    "../Downloads/", 128, dataset=dataset, augment=args.augment
                )
                if not args.eval:
                    mymodule = MyLightningModule(
                        latent=args.latent,
                        arch=arch,
                        optimizer="AdamW",
                        weight_decay=0.0001,
                        lr=0.0001,
                        # momentum=0.9,
                        scheduler="OneCycleLR",
                        input_shape=datamodule.shape,
                        encoder_depth=depth,
                        decoder_depth=depth,
                        guidance=guidance,
                    )

                    omega.pl.launch_worker(
                        mymodule,
                        path,
                        datamodule=datamodule,
                        max_epochs=args.epochs,
                        # save_weights_only=True,
                        # checkpoint_every_k_epoch=1,
                        check_val_every_n_epoch=5,
                        precision=16,
                    )
                    del mymodule
                    del datamodule
                else:
                    datamodule.setup()
                    try:
                        model = MyLightningModule.load_from_checkpoint(
                            path / "checkpoints" / "last.ckpt"
                        )
                    except FileNotFoundError:
                        model = MyLightningModule.load_from_checkpoint(
                            path / "checkpoints" / "last-v1.ckpt"
                        )
                    model = model.cpu()
                    for batch in datamodule.val_dataloader():
                        rec = model.forward(batch[0])
                        break
                    utils.plot_images(rec[:16])
                    asdf
                    score[guidance] = utils.linear_evaluation(
                        MyLightningModule, path, datamodule
                    )
                    print(score)
                asdf
