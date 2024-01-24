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
import omega.pl

plt.style.use("seaborn-v0_8-poster")


class MyDataset(pl.LightningDataModule):
    def __init__(
        self, path, batch_size, train_transform, val_transform, corruption_label
    ):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.corruption_label = corruption_label

    def setup(self, stage: str = None):
        if hasattr(self, "train"):
            return
        dataset = aidatasets.images.CIFAR10(path=self.path).download().load()

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
        self.probe = torch.nn.Linear(self.latent, 10)

    def create_metrics(self):
        self.train_error = torchmetrics.MeanSquaredError()
        self.val_metric = torchmetrics.MeanSquaredError()

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch):
        return self.model.encoder(batch[0]), batch[1]

    def compute_loss(self, batch, batch_idx):
        x, y = batch
        h = self.model.encoder(x)
        xrec = self.model.decoder(h)
        rec_loss = torch.nn.functional.mse_loss(xrec, x)
        self.log("rec_loss", rec_loss)
        if 0:  # batch_idx == 0:
            self.log_images(f"input{self.training}", x, normalize=True, scale_each=True)
            self.log_images(
                f"rec{self.training}", xrec, normalize=True, scale_each=True
            )
        return rec_loss + torch.nn.functional.cross_entropy(self.probe(h), y)


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
                path = args.path / "guidance" / f"ae_{dataset.upper()}_{arch}_{depth}"
                datamodule = MyDataset(
                    "../Downloads/",
                    256,
                    train_transform=transforms.ToTensor(),
                    val_transform=transforms.ToTensor(),
                    corruption_label=args.corruption,
                )
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

                    omega.pl.launch_worker(
                        mymodule,
                        path,
                        datamodule=datamodule,
                        max_epochs=args.epochs,
                        save_weights_only=True,
                        check_val_every_n_epoch=300,
                    )
                else:
                    datamodule.setup()
                    score = utils.linear_evaluation(MyLightningModule, path, datamodule)
                    print(score)
