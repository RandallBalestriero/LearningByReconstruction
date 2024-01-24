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
    def __init__(self, latent, encoder_depth, decoder_depth):
        super().__init__()
        widths = np.linspace(32, latent, encoder_depth).astype("int")
        widths = [3] + list(widths)
        encoder = []
        for fin, fout in zip(widths[:-2], widths[1:-1]):
            encoder.extend([nn.Conv2d(fin, fout, 3, stride=2, padding=1), nn.ReLU()])
        encoder.append(nn.Conv2d(fout, latent, 3, stride=2, padding=1))
        self.encoder = nn.Sequential(*encoder)

        widths = widths[::-1]
        decoder = []
        for fin, fout in zip(widths[:-2], widths[1:-1]):
            decoder.extend(
                [
                    nn.ConvTranspose2d(
                        fin, fout, 3, stride=2, padding=1, output_padding=1
                    ),
                    nn.ReLU(),
                ]
            )
        decoder.append(
            nn.ConvTranspose2d(fout, 3, 3, stride=2, padding=1, output_padding=1)
        )
        decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

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
        decoder.append(nn.Sigmoid())
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
            self.model = DCAE(
                self.latent,
                encoder_depth=self.encoder_depth,
                decoder_depth=self.decoder_depth,
            )
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

    def forward(self, x):
        return self.model(x)

    def predict_step(self, x):
        return self.forward(x), x

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
    parser.add_argument("--arch", type=lambda x: x.split(","))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dataset", type=lambda x: x.split(","))
    args = parser.parse_args()

    for dataset in args.dataset:
        for arch in args.arch:
            for depth in [3, 5]:
                path = args.path / f"ae_{dataset.upper()}_{arch}_{depth}"
                mymodule = MyLightningModule(
                    latent=512,
                    arch=arch,
                    optimizer="AdamW",
                    weight_decay=0.001,
                    lr=0.001,
                    scheduler="OneCycleLR",
                    input_shape=aidatasets.images.__dict__[dataset].SHAPE,
                    encoder_depth=depth,
                    decoder_depth=depth,
                )

                datamodule = aidatasets.utils.dataset_to_lightning(
                    aidatasets.images.__dict__[dataset].load,
                    "../../Downloads/",
                    256,
                    num_workers=31,
                    train_transform=transforms.ToTensor(),
                    val_transform=transforms.ToTensor(),
                )
                if not args.plot:
                    omega.pl.launch_worker(
                        mymodule,
                        path,
                        datamodule=datamodule,
                        max_epochs=args.epochs,
                        checkpoint_every_k_epoch=1,
                        save_weights_only=True,
                        check_val_every_n_epoch=300,
                    )
                else:
                    import matplotlib

                    with torch.no_grad():
                        cmap = matplotlib.cm.get_cmap("coolwarm")
                        datamodule.setup(0)
                        C = None
                        Shats = []
                        loader = datamodule.train_dataloader(
                            persistent_workers=False,
                            num_workers=1,
                            shuffle=False,
                            drop_last=False,
                        )
                        for e in [
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            8,
                            12,
                            20,
                            30,
                            49,
                            # 80,
                            # 110,
                            # 150,
                            # 200,
                            # 299,
                        ]:
                            print(e)
                            model = MyLightningModule.load_from_checkpoint(
                                path / "checkpoints" / f"epoch{e:06}.ckpt"
                            )
                            model.eval()
                            preds = []
                            for batch in loader:
                                preds.append(
                                    model.predict_step(batch[0].cuda(non_blocking=True))
                                )
                                preds[-1] = preds[-1][0].to(
                                    "cpu", non_blocking=True
                                ), preds[-1][1].to("cpu", non_blocking=True)

                            if C is None:
                                X = (
                                    torch.cat([p[1] for p in preds])
                                    .flatten(1)
                                    .cpu()
                                    .numpy()
                                )[::3]
                                for i in range(len(preds)):
                                    preds[i] = (preds[i][0], 1)
                                print("START")
                                X = X.T @ X
                                print("XTX")
                                S, C = np.linalg.eigh(X)
                                print("EIGH")
                                del X
                            Xhat = (
                                torch.cat([p[0] for p in preds])
                                .flatten(1)
                                .cpu()
                                .numpy()
                            )[::2]
                            Shats.append(np.square(Xhat @ C).sum(0))
                        Shats = np.stack(Shats)
                        fig, ax = plt.subplots(figsize=(8, 5))
                        for i, t in enumerate(np.linspace(0.1, 0.9, Shats.shape[0])):
                            plt.semilogy(Shats[i], color=cmap(1 - t))
                        plt.semilogy(S, c="k", linewidth=3)
                        plt.xlim(0, Shats.shape[1])
                        plt.xlabel("k")
                        plt.ylabel("eigenvalue (log-scale)")
                        axins = ax.inset_axes(
                            [0.05, 0.46, 0.4, 0.5],
                            xlim=(int(0.8 * Shats.shape[1]), Shats.shape[1]),
                            ylim=(S[int(0.7 * Shats.shape[1])], S[-3]),
                        )
                        for i, t in enumerate(np.linspace(0.1, 0.9, Shats.shape[0])):
                            axins.semilogy(Shats[i], color=cmap(1 - t))
                        axins.semilogy(S, c="k", linewidth=3)
                        axins.set_yticks([])
                        axins.set_xticks([])
                        fig.text(
                            0.23,
                            0.05,
                            "<- perception",
                            color="tab:blue",
                            horizontalalignment="center",
                            fontsize=13,
                        )
                        fig.text(
                            0.9,
                            0.05,
                            "reconstruction ->",
                            color="tab:red",
                            horizontalalignment="center",
                            fontsize=13,
                        )

                        ax.indicate_inset_zoom(axins, edgecolor="black")
                        plt.tight_layout()
                        plt.savefig(f"spectrum_evolution_{dataset}_{arch}_{depth}.png")
                        plt.close()
