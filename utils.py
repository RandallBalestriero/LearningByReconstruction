import omega
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from torch import nn


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


def linear_evaluation(module, path, dataset):
    try:
        model = module.load_from_checkpoint(path / "checkpoints" / "last.ckpt")
    except FileNotFoundError:
        model = module.load_from_checkpoint(path / "checkpoints" / "last-v1.ckpt")
    model.eval()
    original = dataset.train.transform
    dataset.train.transform = dataset.val.transform
    with torch.no_grad():
        preds = omega.pl.predict(model, dataset.train_dataloader())
        X = torch.cat([p[0] for p in preds]).numpy()
        y = torch.cat([p[1] for p in preds]).numpy()
    fitter = LogisticRegression(verbose=2, C=np.inf, random_state=0, max_iter=1000)
    scaler = StandardScaler(copy=False)
    y_encoder = LabelEncoder().fit(y)
    fitter.fit(scaler.fit_transform(X), y_encoder.transform(y))
    with torch.no_grad():
        preds = omega.pl.predict(model, dataset.val_dataloader())
        X = torch.cat([p[0] for p in preds]).numpy()
        y = torch.cat([p[1] for p in preds]).numpy()

    score = fitter.score(scaler.transform(X), y_encoder.transform(y))
    dataset.train.transform = original
    return score
