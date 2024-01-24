from pathlib import Path
from omega import utils
import omega

from torchvision import transforms
import torch
import torchmetrics
import utils
import lightning.pytorch as pl

torch.set_float32_matmul_precision("medium")
import matplotlib.pyplot as plt
import aidatasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

plt.style.use("seaborn-v0_8-poster")


class MyDataset(pl.LightningDataModule):
    def __init__(
        self, level, path, batch_size, train_transform, val_transform, corruption_label
    ):
        super().__init__()
        self.path = path
        self.level = level
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.corruption_label = corruption_label

    def setup(self, stage: str):
        if hasattr(self, "train"):
            return
        dataset = aidatasets.images.CIFAR10C(path=self.path).download().load()
        dataset["X"] = dataset["X"][dataset["corruption_level"] == self.level]
        dataset["y"] = dataset["y"][dataset["corruption_level"] == self.level]
        dataset["corruption_name"] = dataset["corruption_name"][
            dataset["corruption_level"] == self.level
        ]
        train, val = train_test_split(range(len(dataset["X"])))

        self.train = aidatasets.utils.TensorDataset(
            dataset["X"][train],
            dataset["corruption_name"][train]
            if self.corruption_label
            else dataset["y"][train],
            transform=self.train_transform,
        )
        self.train = aidatasets.utils.TensorDataset(
            dataset["X"][val],
            dataset["corruption_name"][train]
            if self.corruption_label
            else dataset["y"][train],
            transform=self.val_transform,
        )

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

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size)


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

    def create_metrics(self):
        self.train_error = torchmetrics.MeanSquaredError()
        self.val_metric = torchmetrics.MeanSquaredError()

    def forward(self, x):
        return self.model(x)

    def predict_step(self, x):
        return self.forward(x), x

    def compute_loss(self, batch, batch_idx):
        x, _ = batch
        xrec = self.model(x)
        return torch.nn.functional.mse_loss(xrec, x)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path
    import omega
    import aidatasets

    parser = ArgumentParser()
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--arch", type=lambda x: x.split(","), default=["MLPAE"])
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dataset", type=lambda x: x.split(","), default=["CIFAR10C"])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--corruption", action="store_true")
    args = parser.parse_args()

    for dataset in args.dataset:
        for arch in args.arch:
            for depth in [3, 5]:
                for level in range(1, 5):
                    path = args.path / f"ae_{dataset.upper()}_{arch}_{depth}_{level}"
                    if not args.eval:
                        mymodule = MyLightningModule(
                            latent=512,
                            arch=arch,
                            optimizer="AdamW",
                            weight_decay=0.001,
                            lr=0.001,
                            scheduler="OneCycleLR",
                            input_shape=(3, 32, 32),
                            encoder_depth=depth,
                            decoder_depth=depth,
                        )

                        datamodule = MyDataset(
                            level,
                            "../Downloads/",
                            256,
                            train_transform=transforms.ToTensor(),
                            val_transform=transforms.ToTensor(),
                            corruption_label=args.corruption,
                        )

                        omega.pl.launch_worker(
                            mymodule,
                            path,
                            datamodule=datamodule,
                            max_epochs=args.epochs,
                            save_weights_only=True,
                            check_val_every_n_epoch=300,
                        )
                    else:
                        score = utils.linear_evaluation(
                            MyLightningModule, args.path, datamodule
                        )
