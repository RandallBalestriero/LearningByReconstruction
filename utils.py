import omega
import torch
import numpy as np
from torch import nn
import torch


def generalized_solution(
    K: int, X: torch.Tensor, noise=None, image_shape=None, patch_shape=None
):
    from scipy.linalg import eigh

    if noise is None:
        M1 = X
        M2 = X @ X.T
    elif "gaussian" in noise:
        std = float(noise.split("-")[1])
        M1 = X
        M2 = X @ X.T + X.shape[1] * std**2 * np.eye(X.shape[0])
    elif "mask" in noise:
        with torch.no_grad():
            idx = torch.arange(X.shape[0])
            idx = idx.reshape(image_shape).unsqueeze(0)
            patches = torch.nn.functional.unfold(
                idx.float(), patch_shape, stride=patch_shape
            ).long()
        patches = patches[0].T.numpy()
        p = float(noise.split("-")[1])
        M = np.full((X.shape[0], X.shape[0]), (1 - p) * (1 - p))
        for pa in patches:
            idx = np.meshgrid(pa, pa, copy=False)
            M[idx[0].flatten(), idx[1].flatten()] = 1 - p
        M1 = X * (1 - p)
        M2 = (X @ X.T) * M
    _, Vstar = eigh(
        ((M1 @ X.T) @ (X @ M1.T)),
        b=M2,
        subset_by_index=[X.shape[0] - K, X.shape[0] - 1],
    )
    print(Vstar.shape)
    Wstar = np.linalg.solve(Vstar.T @ M2 @ Vstar, Vstar.T @ M1 @ X.T)
    return Wstar, Vstar


def bilinear_solution(X, Y, K, l):

    D, N = X.shape
    P_B, sigma_B, _ = np.linalg.svd(X, full_matrices=False)
    np.square(sigma_B, out=sigma_B)
    if D < N:
        XXT = X @ X.T  # this is D x D
        XYT = X @ Y.T  # this is D x D
        sigma_A, P_A = np.linalg.eigh(XYT @ XYT.T + l * XXT @ XXT)
    else:
        M = Y.T @ Y + l * X.T @ X  # this is N x N
        sigma_M, P_M = np.linalg.eigh(M)
        np.clip(sigma_M, 0, None, out=sigma_M)
        S = (np.sqrt(sigma_M)[:, None] * P_M.T) @ X.T  # this is N x D
        _, sigma_A, V_At = np.linalg.svd(S, full_matrices=False)
        sigma_A = np.square(sigma_A, out=sigma_A)
        P_A = V_At.T

    assert P_A.shape == (D, min(D, N))
    assert sigma_A.shape == (min(D, N),)
    assert sigma_B.shape == (min(D, N),)
    assert P_B.shape == (D, min(D, N))

    np.clip(sigma_A, 0, None, out=sigma_A)
    sinv = np.nan_to_num(1 / (1e-6 + np.sqrt(sigma_B)))
    H = np.diag(np.sqrt(sigma_A)) @ P_A.T @ P_B @ np.diag(sinv)
    _, sigma_H, P_Ht = np.linalg.svd(H, full_matrices=False)
    P_H = P_Ht[:K].T
    Vstar = P_B @ np.diag(sinv) @ P_H
    if D < N:
        Wstar = np.linalg.solve(Vstar.T @ XXT @ Vstar, Vstar.T @ XYT)
        Zstar = np.linalg.solve(Vstar.T @ XXT @ Vstar, Vstar.T @ XXT)
    else:
        M = (Vstar.T @ X) @ (X.T @ Vstar)
        Wstar = np.linalg.solve(M, (Vstar.T @ X) @ Y.T)
        Zstar = np.linalg.solve(M, (Vstar.T @ X) @ X.T)

    return Wstar, Zstar, Vstar


def get_configs(arch="resnet50"):
    # True or False means wether to use BottleNeck

    if arch == "resnet18":
        return [2, 2, 2, 2], False
    elif arch == "resnet34":
        return [3, 4, 6, 3], False
    elif arch == "resnet50":
        return [3, 4, 6, 3], True
    elif arch == "resnet101":
        return [3, 4, 23, 3], True
    elif arch == "resnet152":
        return [3, 8, 36, 3], True
    else:
        raise ValueError("Undefined model")


class ResNetAutoEncoder(nn.Module):
    def __init__(self, configs, bottleneck, latent_shape, latent_target):
        super(ResNetAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            ResNetEncoder(configs=configs, bottleneck=bottleneck),
            nn.Flatten(),
            nn.Linear(np.prod(latent_shape), latent_target),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_target, np.prod(latent_shape)),
            nn.Unflatten(1, latent_shape),
            ResNetDecoder(configs=configs[::-1], bottleneck=bottleneck),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class ResNet(nn.Module):
    def __init__(self, configs, bottleneck=False, num_classes=1000):
        super(ResNet, self).__init__()

        self.encoder = ResNetEncoder(configs, bottleneck)

        self.avpool = nn.AdaptiveAvgPool2d((1, 1))

        if bottleneck:
            self.fc = nn.Linear(in_features=2048, out_features=num_classes)
        else:
            self.fc = nn.Linear(in_features=512, out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)

        x = self.avpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


class ResNetEncoder(nn.Module):
    def __init__(self, configs, bottleneck=False):
        super(ResNetEncoder, self).__init__()

        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configued")

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        if bottleneck:
            self.conv2 = EncoderBottleneckBlock(
                in_channels=64,
                hidden_channels=64,
                up_channels=256,
                layers=configs[0],
                downsample_method="pool",
            )
            self.conv3 = EncoderBottleneckBlock(
                in_channels=256,
                hidden_channels=128,
                up_channels=512,
                layers=configs[1],
                downsample_method="conv",
            )
            self.conv4 = EncoderBottleneckBlock(
                in_channels=512,
                hidden_channels=256,
                up_channels=1024,
                layers=configs[2],
                downsample_method="conv",
            )
            self.conv5 = EncoderBottleneckBlock(
                in_channels=1024,
                hidden_channels=512,
                up_channels=2048,
                layers=configs[3],
                downsample_method="conv",
            )

        else:
            self.conv2 = EncoderResidualBlock(
                in_channels=64,
                hidden_channels=64,
                layers=configs[0],
                downsample_method="pool",
            )
            self.conv3 = EncoderResidualBlock(
                in_channels=64,
                hidden_channels=128,
                layers=configs[1],
                downsample_method="conv",
            )
            self.conv4 = EncoderResidualBlock(
                in_channels=128,
                hidden_channels=256,
                layers=configs[2],
                downsample_method="conv",
            )
            self.conv5 = EncoderResidualBlock(
                in_channels=256,
                hidden_channels=512,
                layers=configs[3],
                downsample_method="conv",
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


class ResNetDecoder(nn.Module):
    def __init__(self, configs, bottleneck=False):
        super(ResNetDecoder, self).__init__()

        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configued")

        if bottleneck:
            self.conv1 = DecoderBottleneckBlock(
                in_channels=2048,
                hidden_channels=512,
                down_channels=1024,
                layers=configs[0],
            )
            self.conv2 = DecoderBottleneckBlock(
                in_channels=1024,
                hidden_channels=256,
                down_channels=512,
                layers=configs[1],
            )
            self.conv3 = DecoderBottleneckBlock(
                in_channels=512,
                hidden_channels=128,
                down_channels=256,
                layers=configs[2],
            )
            self.conv4 = DecoderBottleneckBlock(
                in_channels=256, hidden_channels=64, down_channels=64, layers=configs[3]
            )

        else:
            self.conv1 = DecoderResidualBlock(
                hidden_channels=512, output_channels=256, layers=configs[0]
            )
            self.conv2 = DecoderResidualBlock(
                hidden_channels=256, output_channels=128, layers=configs[1]
            )
            self.conv3 = DecoderResidualBlock(
                hidden_channels=128, output_channels=64, layers=configs[2]
            )
            self.conv4 = DecoderResidualBlock(
                hidden_channels=64, output_channels=64, layers=configs[3]
            )

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=3,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1,
                bias=False,
            ),
        )

        self.gate = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x


class EncoderResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, layers, downsample_method="conv"):
        super(EncoderResidualBlock, self).__init__()

        if downsample_method == "conv":
            for i in range(layers):
                if i == 0:
                    layer = EncoderResidualLayer(
                        in_channels=in_channels,
                        hidden_channels=hidden_channels,
                        downsample=True,
                    )
                else:
                    layer = EncoderResidualLayer(
                        in_channels=hidden_channels,
                        hidden_channels=hidden_channels,
                        downsample=False,
                    )

                self.add_module("%02d EncoderLayer" % i, layer)

        elif downsample_method == "pool":
            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module("00 MaxPooling", maxpool)

            for i in range(layers):
                if i == 0:
                    layer = EncoderResidualLayer(
                        in_channels=in_channels,
                        hidden_channels=hidden_channels,
                        downsample=False,
                    )
                else:
                    layer = EncoderResidualLayer(
                        in_channels=hidden_channels,
                        hidden_channels=hidden_channels,
                        downsample=False,
                    )

                self.add_module("%02d EncoderLayer" % (i + 1), layer)

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)

        return x


class EncoderBottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        up_channels,
        layers,
        downsample_method="conv",
    ):
        super(EncoderBottleneckBlock, self).__init__()

        if downsample_method == "conv":
            for i in range(layers):
                if i == 0:
                    layer = EncoderBottleneckLayer(
                        in_channels=in_channels,
                        hidden_channels=hidden_channels,
                        up_channels=up_channels,
                        downsample=True,
                    )
                else:
                    layer = EncoderBottleneckLayer(
                        in_channels=up_channels,
                        hidden_channels=hidden_channels,
                        up_channels=up_channels,
                        downsample=False,
                    )

                self.add_module("%02d EncoderLayer" % i, layer)

        elif downsample_method == "pool":
            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module("00 MaxPooling", maxpool)

            for i in range(layers):
                if i == 0:
                    layer = EncoderBottleneckLayer(
                        in_channels=in_channels,
                        hidden_channels=hidden_channels,
                        up_channels=up_channels,
                        downsample=False,
                    )
                else:
                    layer = EncoderBottleneckLayer(
                        in_channels=up_channels,
                        hidden_channels=hidden_channels,
                        up_channels=up_channels,
                        downsample=False,
                    )

                self.add_module("%02d EncoderLayer" % (i + 1), layer)

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)

        return x


class DecoderResidualBlock(nn.Module):
    def __init__(self, hidden_channels, output_channels, layers):
        super(DecoderResidualBlock, self).__init__()

        for i in range(layers):
            if i == layers - 1:
                layer = DecoderResidualLayer(
                    hidden_channels=hidden_channels,
                    output_channels=output_channels,
                    upsample=True,
                )
            else:
                layer = DecoderResidualLayer(
                    hidden_channels=hidden_channels,
                    output_channels=hidden_channels,
                    upsample=False,
                )

            self.add_module("%02d EncoderLayer" % i, layer)

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)

        return x


class DecoderBottleneckBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, down_channels, layers):
        super(DecoderBottleneckBlock, self).__init__()

        for i in range(layers):
            if i == layers - 1:
                layer = DecoderBottleneckLayer(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    down_channels=down_channels,
                    upsample=True,
                )
            else:
                layer = DecoderBottleneckLayer(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    down_channels=in_channels,
                    upsample=False,
                )

            self.add_module("%02d EncoderLayer" % i, layer)

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)

        return x


class EncoderResidualLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, downsample):
        super(EncoderResidualLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.Sequential(nn.ReLU(inplace=True))

    def forward(self, x):
        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        x = self.relu(x)

        return x


class EncoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, up_channels, downsample):
        super(EncoderBottleneckLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.weight_layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=up_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=up_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=up_channels,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=up_channels),
            )
        elif in_channels != up_channels:
            self.downsample = None
            self.up_scale = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=up_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=up_channels),
            )
        else:
            self.downsample = None
            self.up_scale = None

        self.relu = nn.Sequential(nn.ReLU(inplace=True))

    def forward(self, x):
        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        elif self.up_scale is not None:
            identity = self.up_scale(identity)

        x = x + identity

        x = self.relu(x)

        return x


class DecoderResidualLayer(nn.Module):
    def __init__(self, hidden_channels, output_channels, upsample):
        super(DecoderResidualLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=hidden_channels,
                    out_channels=output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=output_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=hidden_channels,
                    out_channels=output_channels,
                    kernel_size=1,
                    stride=2,
                    output_padding=1,
                    bias=False,
                ),
            )
        else:
            self.upsample = None

    def forward(self, x):
        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity

        return x


class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, down_channels, upsample):
        super(DecoderBottleneckLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        self.weight_layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        if upsample:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=hidden_channels,
                    out_channels=down_channels,
                    kernel_size=1,
                    stride=2,
                    output_padding=1,
                    bias=False,
                ),
            )
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=down_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=down_channels,
                    kernel_size=1,
                    stride=2,
                    output_padding=1,
                    bias=False,
                ),
            )
        elif in_channels != down_channels:
            self.upsample = None
            self.down_scale = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=down_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            )
        else:
            self.upsample = None
            self.down_scale = None

    def forward(self, x):
        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.upsample is not None:
            identity = self.upsample(identity)
        elif self.down_scale is not None:
            identity = self.down_scale(identity)

        x = x + identity

        return x


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
    with torch.no_grad():
        preds = omega.pl.predict(model, dataset.train_dataloader())
        print(np.mean(preds))


def plot_images(tensor, name=None):
    import torchvision
    import matplotlib.pyplot as plt

    im = torchvision.utils.make_grid(tensor, normalize=True, scale_each=True)
    nrows = max(1, len(tensor) // 8)
    plt.figure(figsize=((len(tensor) * 2) / nrows, 2 * nrows))
    plt.imshow(im.permute(1, 2, 0), aspect="auto")
    plt.axis("off")
    plt.tight_layout()
    if name is not None:
        plt.savefig(name)
        plt.close()
    else:
        plt.show()
