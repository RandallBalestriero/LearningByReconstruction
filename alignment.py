import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import utils

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

plt.style.use("seaborn-v0_8-poster")


def Xy_to_errors(X, y, noise=None, image_shape=None, patch_shape=None):

    if X.shape[1] < 4000:
        _, S = utils.generalized_solution(
            X.shape[1],
            X.T,
            noise=noise,
            image_shape=image_shape,
            patch_shape=patch_shape,
        )
        pS = X @ S[:, ::-1]
        print("Doing QR on", pS.shape)
        pS = np.linalg.qr(pS)[0]
    else:
        X += np.random.randn(*X.shape) * 0.01 * X.std()
        print(X.shape, y.shape)
        U, S = np.linalg.eigh(X.T @ X)
        pS = X @ S
        del X
        pS /= np.sqrt(U)
        pS = pS[:, ::-1]

    numer = y @ (y.T @ pS) / y.shape[0]
    np.square(numer, out=numer)
    numer = numer.sum(0)
    np.cumsum(numer, out=numer)
    np.sqrt(numer, out=numer)
    numer /= 0.01 * numer[-1]
    return numer


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path
    import numpy as np
    import aidatasets
    from tqdm import tqdm

    parser = ArgumentParser()
    parser.add_argument("--path", type=Path, default=None)
    args = parser.parse_args()

    K = 256

    for dname in ["SVHN", "CIFAR10"]:
        dataset = aidatasets.images.__dict__[dname]("../Downloads/").download().load()
        X, y = dataset["train_X"].astype("float64"), dataset["train_y"].astype(
            "float64"
        )
        print(X.shape)
        X = X.reshape((X.shape[0], -1))
        X += np.random.randn(*X.shape) * 0.001
        X -= X.mean(0)
        X /= X.std()
        y = OneHotEncoder(sparse_output=False).fit_transform(y[:, None])

        probas = np.linspace(0, 0.99, 10)
        colors = [
            matplotlib.colormaps["cool"](t) for t in np.linspace(0, 1, len(probas))
        ]
        for j, k in enumerate([1, 2, 4]):
            numer = np.zeros((len(probas), X.shape[1]))
            for i, std in enumerate(probas):
                numer[i] = Xy_to_errors(
                    X,
                    y,
                    noise=f"mask-{std}",
                    image_shape=dataset.image_shape,
                    patch_shape=(k, k),
                )
            numer = np.round(100 * (numer - numer[0]) / numer[0], 2)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            for c in range(len(probas)):
                ax.plot(
                    range(32, X.shape[1]),
                    numer[c, 32:],
                    c=colors[c],
                    linewidth=2,
                    alpha=0.8,
                )
            ax.set_xlim(32, X.shape[1])
            ax.axhline(0, c="k", linewidth=3)
            ax.set_ylabel("impact on sup. perf. (%)")
            ax.set_xlabel("embedding dim. (K)")
            plt.tight_layout()
            plt.savefig(f"generalized_alignment_{dname}_{k}.png")
            plt.close()
    asdf

    fig, axs = plt.subplots(2, 1, figsize=(7, 9))

    for dname in [
        "ArabicCharacters",
        "ArabicDigits",
        "SVHN",
        "FashionMNIST",
        "CIFAR10",
        "CIFAR100",
        # "emnist",
        # "tinyimagenet",
    ]:
        dataset = aidatasets.images.__dict__[dname]("../Downloads/").download().load()
        if dname == "tinyimagenet":
            dataset["train_X"] = np.stack(
                [np.array(x) for x in dataset["train"]["X"][::2]]
            )
            dataset["train_y"] = (
                OrdinalEncoder()
                .fit_transform(dataset["train"]["y"][::2, None])
                .squeeze()
            )
        X, y = dataset["train_X"].astype("float64"), dataset["train_y"].astype(
            "float64"
        )
        X = X.reshape((X.shape[0], -1))
        X += np.random.randn(*X.shape) * 0.01
        y = OneHotEncoder(sparse_output=False).fit_transform(y[:, None])
        numer = Xy_to_errors(X, y)
        axs[0].plot(numer[:K], label=dname, linewidth=3, alpha=0.7)
        axs[1].plot(np.linspace(0, 100, len(numer)), numer, label=dname)

    for ax in axs:
        ax.set_ylim(0, 100)
        ax.set_ylabel("task alignment (%)")
    axs[0].set_xlim(1, K)
    axs[1].set_xlim(0, 100)
    axs[0].set_xlabel(r"latent dimension ($K$)")

    axs[1].set_xlabel("latent dimension (% of input dimension)")
    plt.legend(fontsize=15, ncol=2)
    plt.tight_layout()
    plt.savefig("exact_alignment.png")
    plt.close()
