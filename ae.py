from pathlib import Path
from omega import utils
import omega

from torchvision import transforms
import torch
import torchmetrics
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import os
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import ndimage

torch.set_float32_matmul_precision("medium")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-poster")


class DCAE(nn.Module):
    def __init__(self, latent):
        super().__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=1, output_padding=1
            ),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class MLPAE(nn.Module):
    def __init__(self, inputs, latent, encoder_depth, decoder_depth):
        super().__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        widths = np.linspace(latent, np.prod(inputs), encoder_depth).astype("int")[::-1]
        print(widths)
        encoder = [nn.Flatten()]
        fout = widths[0]
        for fin, fout in zip(widths[:-2], widths[1:-1]):
            encoder.append(
                nn.Sequential(
                    nn.Linear(fin, fout, bias=False), nn.BatchNorm1d(fout), nn.ReLU()
                )
            )
        encoder.append(nn.Linear(fout, latent))
        self.encoder = nn.Sequential(*encoder)

        widths = np.linspace(latent, np.prod(inputs), decoder_depth).astype("int")
        print(widths)
        decoder = []
        fout = widths[0]
        for fin, fout in zip(widths[:-2], widths[1:-1]):
            decoder.append(
                nn.Sequential(
                    nn.Linear(fin, fout, bias=False), nn.BatchNorm1d(fout), nn.ReLU()
                )
            )
        decoder.append(nn.Linear(fout, np.prod(inputs)))
        decoder.append(nn.Unflatten(1, inputs))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class MyLightningModule(omega.pl.Module):
    def create_modules(self):
        if self.arch == "DCAE":
            self.model = DCAE(self.latent)
        elif self.arch == "MLPAE":
            self.model = MLPAE(
                self.input_shape,
                self.latent,
                encoder_depth=self.encoder_depth,
                decoder_depth=self.decoder_depth,
            )

    def create_metrics(self):
        self.train_error = torchmetrics.MeanSquaredError()
        self.val_metric = torchmetrics.MeanSquaredError()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model.encode(batch[0]), batch[1]

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch, batch_idx):
        x, _ = batch
        xrec = self.model(x)
        if 0:  # batch_idx == 0:
            self.log_images(f"input{self.training}", x, normalize=True, scale_each=True)
            self.log_images(
                f"rec{self.training}", xrec, normalize=True, scale_each=True
            )
        return torch.nn.functional.mse_loss(xrec, x)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path
    import omega
    import aidatasets

    parser = ArgumentParser()
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--arch", choices=["DCAE", "MLPAE"])
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.plot:
        import matplotlib.pyplot as plt
        import glob
        import sys
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        for latent in [128]:
            fig, axs = plt.subplots(3, 2, sharex="all", sharey="all", figsize=(8, 8))
            for i, (ax, dname) in enumerate(
                zip(
                    axs.reshape(-1),
                    [
                        # "emnist",
                        "fashionmnist",
                        "SVHN",
                        "arabic_digits",
                        "arabic_characters",
                    ],
                )
            ):
                scores = []
                for encoder_depth in range(2, 13):
                    for decoder_depth in range(2, 13):
                        fname = f"{latent}_{encoder_depth}_{decoder_depth}/score.txt"
                        scores.append(
                            (args.path / f"ae_{dname.upper()}_MLP" / fname).read_text()
                        )
                scores = (
                    np.asarray(scores).reshape((len(range(2, 13)), -1)).astype("float")
                    * 100
                )
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)

                im = ax.imshow(
                    scores,
                    extent=[0, 1, 0, 1],
                    origin="lower",
                    aspect="auto",
                    cmap="coolwarm",
                )
                ax.set_xticks(
                    np.linspace(0, 1 - 1 / scores.shape[1], scores.shape[1])
                    + 0.5 / scores.shape[1],
                    range(2, 2 + scores.shape[1]),
                )
                ax.set_yticks(
                    np.linspace(0, 1 - 1 / scores.shape[0], scores.shape[0])
                    + 0.5 / scores.shape[0],
                    range(2, 2 + scores.shape[0]),
                )
                if i in [2, 3]:
                    ax.set_xlabel("decoder depth")
                if i == 0 or i == 2:
                    ax.set_ylabel("encoder depth")
                ax.set_title(dname)
                cbar = fig.colorbar(im, cax=cax, orientation="vertical")
                cbar.ax.yaxis.set_tick_params(pad=1)
            plt.subplots_adjust(0.1, 0.1, 0.95, 0.95, 0.2, 0.2)
            plt.savefig("encoder_decoder_depth.png")
            plt.close()
        sys.exit()
    for dname in [
        # "emnist",
        "arabic_characters",
        "arabic_digits",
        # "mnist",
        "svhn",
        "fashionmnist",
        "cifar10",
        "cifar100",
    ]:
        for latent in [128]:
            for encoder_depth in range(2, 13):
                for decoder_depth in range(2, 13):
                    path = (
                        args.path
                        / f"ae_{dname.upper()}_MLP"
                        / f"{latent}_{encoder_depth}_{decoder_depth}"
                    )

                    if (path / "score.txt").is_file():
                        continue
                    mymodule = MyLightningModule(
                        latent=latent,
                        arch=args.arch,
                        optimizer="AdamW",
                        weight_decay=0.001,
                        lr=0.001,
                        scheduler="OneCycleLR",
                        input_shape=aidatasets.images.__dict__[dname].SHAPE,
                        encoder_depth=encoder_depth,
                        decoder_depth=decoder_depth,
                    )

                    dataset = aidatasets.utils.dataset_to_lightning(
                        aidatasets.images.__dict__[dname].load,
                        "../../Downloads/",
                        256,
                        num_workers=31,
                        train_transform=transforms.Compose(
                            [transforms.ToTensor(), transforms.RandomHorizontalFlip()]
                        ),
                        val_transform=transforms.ToTensor(),
                    )
                    if dname == "arabic_characters" or dname == "arabic_digits":
                        epochs = 120
                    else:
                        epochs = 60
                    omega.pl.launch_worker(
                        mymodule,
                        path,
                        datamodule=dataset,
                        max_epochs=epochs,
                    )
                    try:
                        model = MyLightningModule.load_from_checkpoint(
                            path / "checkpoints" / "last.ckpt"
                        )
                    except FileNotFoundError:
                        model = MyLightningModule.load_from_checkpoint(
                            path / "checkpoints" / "last-v1.ckpt"
                        )
                    model.eval()
                    dataset.train.transform = transforms.ToTensor()
                    with torch.no_grad():
                        preds = omega.pl.predict(model, dataset.train_dataloader())
                        X = torch.cat([p[0] for p in preds]).numpy()
                        y = torch.cat([p[1] for p in preds]).numpy()
                    fitter = LogisticRegression(
                        verbose=2, C=np.inf, random_state=0, max_iter=1000
                    )
                    scaler = StandardScaler(copy=False)
                    fitter.fit(scaler.fit_transform(X), y)
                    with torch.no_grad():
                        preds = omega.pl.predict(model, dataset.val_dataloader())
                        X = torch.cat([p[0] for p in preds]).numpy()
                        y = torch.cat([p[1] for p in preds]).numpy()

                    score = fitter.score(scaler.transform(X), y)
                    (path / "score.txt").write_text(str(score))
