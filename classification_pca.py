from torch import nn
import torch
from torchvision import transforms


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample, self.y[idx]


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()

        self.conv1 = conv_block(inputs[0], 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.conv5 = conv_block(512, 1028, pool=True)
        self.res3 = nn.Sequential(conv_block(1028, 1028), conv_block(1028, 1028))

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(1028, outputs[0])
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out


class MLP(nn.Module):
    def __init__(self, inputs, outputs, depth):
        super().__init__()
        widths = np.linspace(np.prod(outputs), np.prod(inputs), depth).astype("int")[
            ::-1
        ]
        print(widths)
        encoder = [nn.Flatten()]
        fout = widths[0]
        for fin, fout in zip(widths[:-2], widths[1:-1]):
            encoder.append(
                nn.Sequential(
                    nn.Linear(fin, fout, bias=False), nn.BatchNorm1d(fout), nn.ReLU()
                )
            )
        encoder.append(nn.Linear(fout, np.prod(outputs)))
        encoder.append(nn.Unflatten(1, outputs))
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


def normalize(X, Xtest):
    X /= 1e-6 + X.std((1, 2, 3), keepdims=True)
    if Xtest is not None:
        Xtest /= 1e-6 + Xtest.std((1, 2, 3), keepdims=True)
        Xtest -= X.mean(0)
    X -= X.mean(0)
    return X, Xtest


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path
    import numpy as np

    parser = ArgumentParser()
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--arch", choices=["R9", "MLP"])
    args = parser.parse_args()

    for da in [1, 0]:
        for dname in [
            "ArabicCharacters",
            "ArabicDigits",
            "SVHN",
            "fashionmnist",
            "CIFAR10",
            "CIFAR100",
            "TinyImagenet",
        ]:
            if dname == "arabic_characters" or dname == "arabic_digits":
                epochs = 100
            else:
                epochs = 50
            if dname == "TinyImagenet":
                # Load X only
                # X = ....
                X = normalize(X, None)[0]
                X = np.transpose(X, (0, 3, 1, 2))
                assert X.shape[1] == 3
                X = X.reshape((X.shape[0], -1))
                C = X.T @ X
                print("C is computed")
                del X
                U, S = np.linalg.eigh(C)
                del C
                S = S.astype("float32")
                print("eigh computed")
                U /= U.sum()
                # Load X, y and Xtest, ytest as numpy arrays in RGB format with (N, H, W, C)
                # X, y = ....
                # Xtest, ytest = ...
            else:
                # Load X, y and Xtest, ytest as numpy arrays in RGB format with (N, H, W, C)
                # X, y = ....
                # Xtest, ytest = ...
            X, Xtest = normalize(X, Xtest)
            X = np.transpose(X, (0, 3, 1, 2))
            assert X.shape[1] == 3
            X = X.reshape((X.shape[0], -1))
            Xtest = np.transpose(Xtest, (0, 3, 1, 2)).reshape((Xtest.shape[0], -1))

            if dname != "TinyImagenet":
                U, S = np.linalg.eigh(X.T @ X)
                U /= U.sum()
                S = S.astype("float32")
            explained = np.cumsum(U[::-1])[::-1]

            for option in ["top", "bottom"]:
                bounds = [explained[-2], explained[explained < 1].max()]
                if option == "top":
                    grid = np.linspace(bounds[0], bounds[1], 8)
                else:
                    grid = np.sqrt(np.sqrt(np.linspace(0, 1, 8)))
                    grid *= bounds[1] - bounds[0]
                    grid += bounds[0]
                for i, pct in enumerate([-1] + list(grid)):
                    if option == "top":
                        if pct == -1:
                            S_ = (
                                S[:, len(explained) // 2 :]
                                @ S[:, len(explained) // 2 :].T
                            )
                            dimensions = int(len(explained) // 2)
                        else:
                            S_ = S[:, explained < pct] @ S[:, explained < pct].T
                            dimensions = int((explained < pct).astype("float").sum())
                    else:
                        if pct == -1:
                            S_ = (
                                S[:, : len(explained) // 2]
                                @ S[:, : len(explained) // 2].T
                            )
                            dimensions = int(len(explained) // 2)
                        else:
                            S_ = S[:, explained >= pct] @ S[:, explained >= pct].T
                            dimensions = int((explained >= pct).astype("float").sum())

                    X_ = np.matmul(X, S_).reshape((-1,) + (3, 32, 32))
                    Xtest_ = np.matmul(Xtest, S_).reshape((-1,) + (3, 32, 32))
                    del S_
                    X_, Xtest_ = normalize(X_, Xtest_)
                    path = (
                        args.path
                        / f"mse_classification_{dname.upper()}_{args.arch}_{da}"
                        / f"{option}_{i}"
                    )

                    trainset = CustomDataset(
                        torch.from_numpy(X_),
                        torch.from_numpy(y).long(),
                        transform=(
                            transforms.Compose(
                                [
                                    transforms.RandomCrop(
                                        64 if dname == "TinyImagenet" else 32,
                                        padding=8 if dname == "TinyImagenet" else 4,
                                        padding_mode="reflect",
                                    ),
                                    transforms.RandomHorizontalFlip(),
                                ]
                            )
                            if da
                            else None
                        ),
                    )
                    trainloader = torch.utils.data.DataLoader(
                        trainset,
                        batch_size=512,
                        shuffle=True,
                        num_workers=31 if dname != "TinyImagenet" else 1,
                        drop_last=True,
                        persistent_workers=True,
                        pin_memory=True,
                    )

                    testset = CustomDataset(
                        torch.from_numpy(Xtest_), torch.from_numpy(ytest).long()
                    )
                    testloader = torch.utils.data.DataLoader(
                        testset, batch_size=512, shuffle=False
                    )
                    
                    # Launch  training with 
                    # outputs=(int(y.max()) + 1,),
                    # arch=args.arch,
                    # optimizer="AdamW",
                    # lr=0.001,
                    # weight_decay=0.001,
                    # scheduler="OneCycleLR",
                    # input_shape=(3, 32, 32),
                    # depth=4,
                    # pct=float(pct),
                    # dimensions=dimensions,
                    # output_dir=path,
                    # train_dataloaders=trainloader,
                    # val_dataloaders=testloader,
                    # max_epochs=epochs,
                    # precision=16,
