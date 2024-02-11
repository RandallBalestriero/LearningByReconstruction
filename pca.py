import tqdm

import torch

torch.set_float32_matmul_precision("medium")
import matplotlib.pyplot as plt
import aidatasets
from torchvision.transforms import v2
import utils

plt.style.use("seaborn-v0_8-poster")


def fast_gram_eigh(X, major="C", unit_test=False):
    """
    compute the eigendecomposition of the Gram matrix:
    - XX.T using column (C) major notation
    - X.T@X using row (R) major notation
    """
    if major == "C":
        X_view = X.T
    else:
        X_view = X

    if X_view.shape[1] < X_view.shape[0]:
        # this case is the usual formula
        U, S = torch.linalg.eigh(X_view.T @ X_view)
    else:
        # in this case we work in the tranpose domain
        U, S = torch.linalg.eigh(X_view @ X_view.T)
        S = X_view.T @ S
        S[:, U > 0] /= torch.sqrt(U[U > 0])

    # ensuring that we have the correct values
    if unit_test:
        Uslow, Sslow = np.linalg.eigh(X_view.T @ X_view)
        assert np.allclose(U, Uslow)
        assert np.allclose(S, Sslow)
    return U, S


train_transform = v2.Compose(
    [
        v2.PILToTensor(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

dataset = aidatasets.images.Imagenette(path="../Downloads/").download().load()

train = aidatasets.utils.TensorDataset(
    dataset["train_X"],
    dataset["train_y"],
    transform=train_transform,
)
del dataset

with torch.no_grad():
    images = torch.zeros(len(train), 3 * 224 * 224)
    print(images.shape)
    for i, (im, y) in tqdm.tqdm(enumerate(train)):
        images[i] = im.flatten()
    m = images.mean(0)
    images.sub_(m)
    h, l = fast_gram_eigh(images, "R")
    # print(C.shape)
    # h, l = torch.linalg.eigh(C)
    h /= h.sum()
    h = h.cumsum(dim=0)
    bottomk = torch.count_nonzero(h < 0.25)
    topk = torch.count_nonzero(h > 0.25)
    pick = [0, 100, 1000, 4000, 5000, 5001]
    pick = [1000, 4000]
    print(bottomk, topk)
    utils.plot_images(
        (images[pick] + m).reshape(-1, 3, 224, 224), "original_teaser.pdf"
    )
    bottom = ((images[pick]) @ l[:, :bottomk]) @ l[:, :bottomk].T + m
    utils.plot_images(bottom.reshape(-1, 3, 224, 224), "bottom_pca_teaser.pdf")
    top = ((images[pick]) @ l[:, -topk:]) @ l[:, -topk:].T + m
    utils.plot_images(top.reshape(-1, 3, 224, 224), "top_pca_teaser.pdf")
    import numpy as np

    pick = np.random.permutation(len(images))[:32]
    utils.plot_images(
        (images[pick] + m).reshape(-1, 3, 224, 224), "original_appendix.pdf"
    )
    bottom = ((images[pick]) @ l[:, :bottomk]) @ l[:, :bottomk].T + m
    utils.plot_images(bottom.reshape(-1, 3, 224, 224), "bottom_pca_appendix.pdf")
    top = ((images[pick]) @ l[:, -topk:]) @ l[:, -topk:].T + m
    utils.plot_images(top.reshape(-1, 3, 224, 224), "top_pca_appendix.pdf")
