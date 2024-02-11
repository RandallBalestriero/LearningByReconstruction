import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import utils

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
N = 256
D = 512
K = 4

X = (torch.randn(D, N) > 0).float()
Y = torch.randn(2, D) @ X / np.sqrt(D)


def test_generalized(noise=None, view=1, steps=30000):

    W = torch.nn.Parameter(torch.randn(K, D))
    V = torch.nn.Parameter(torch.randn(D, K))

    optim = torch.optim.SGD(
        [W, V], lr=0.05, weight_decay=0, momentum=0.9, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps)
    sgd_losses = []
    for i in range(steps):
        optim.zero_grad()
        Xbatch = X.repeat(1, view)
        if noise is not None and "gaussian" in noise:
            std = float(noise.split("-")[1])
            Xbatch = Xbatch + torch.randn(*Xbatch.shape) * std

        loss = torch.nn.functional.mse_loss(
            W.T @ V.T @ Xbatch, X.repeat(1, view), reduction="sum"
        )
        loss.backward()
        optim.step()
        scheduler.step()
        sgd_losses.append(loss.item())
    # if noise is None:
    #     M1 = X
    #     M2 = X @ X.T
    # elif noise == "normal":
    #     M1 = X
    #     M2 = X @ X.T + 0.1**2 * torch.eye(X.size(0))
    # with torch.no_grad():
    #     _, Vstar = eigh(
    #         ((M1 @ X.T) @ (X @ M1.T)).numpy(),
    #         b=M2.numpy(),
    #         subset_by_index=[D - K, D - 1],
    #     )
    #     print(Vstar.shape)
    #     Vstar = torch.from_numpy(Vstar)
    #     Wstar = torch.linalg.solve(Vstar.T @ M2 @ Vstar, Vstar.T @ M1 @ X.T)
    Wstar, Vstar = utils.generalized_solution(K, X.numpy(), noise)
    loss = torch.nn.functional.mse_loss(
        torch.from_numpy(Wstar).float().T @ torch.from_numpy(Vstar).float().T @ Xbatch,
        X.repeat(1, view),
        reduction="sum",
    )
    return np.array(sgd_losses), loss.item()


def test_base(lr, l, steps=50000):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    Z = torch.nn.Parameter(torch.randn(K, D) / np.sqrt(D))
    W = torch.nn.Parameter(torch.randn(K, Y.shape[0]))
    V = torch.nn.Parameter(torch.randn(D, K) / np.sqrt(D))

    optim = torch.optim.Adam([W, V, Z], lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps)
    sgd_losses = []
    for i in range(steps):
        optim.zero_grad()
        loss1 = torch.nn.functional.mse_loss(W.T @ V.T @ X, Y, reduction="sum")
        loss2 = torch.nn.functional.mse_loss(Z.T @ V.T @ X, X, reduction="sum")
        loss = (loss1 + l * loss2) / (D * N)
        loss.backward()
        optim.step()
        scheduler.step()
        sgd_losses.append(loss.item())
    Wstar, Zstar, Vstar = utils.bilinear_solution(X.numpy(), Y.numpy(), K, l)
    Wstar = torch.from_numpy(Wstar).float()
    Vstar = torch.from_numpy(Vstar).float()
    Zstar = torch.from_numpy(Zstar).float()
    loss1 = torch.nn.functional.mse_loss(Wstar.T @ Vstar.T @ X, Y, reduction="sum")
    loss2 = torch.nn.functional.mse_loss(Zstar.T @ Vstar.T @ X, X, reduction="sum")
    loss = (loss1 + l * loss2) / (D * N)
    return np.array(sgd_losses), loss.item()


if __name__ == "__main__":
    # sgd_loss, optimal = test_generalized("gaussian-0.1", 10)

    fig, axs = plt.subplots(1, 4, figsize=(15, 5), sharex="all", sharey="all")
    fig.suptitle("Empirical validation of optimal solution", fontsize=20)
    for i, l in enumerate([0.0, 0.1, 1, 10]):
        optim = [np.inf]
        for lr in [0.1, 0.01, 0.001, 0.0001]:
            sgd_loss, optimal = test_base(lr, l, 1000)
            if np.mean(sgd_loss[-300:]) < np.mean(optim[-300:]):
                optim = sgd_loss
        axs[i].plot(optim - optimal)
        axs[i].set_xlabel("steps (t)", fontsize=20)
        axs[i].set_title(str(np.min(optim - optimal)))
    axs[0].set_yscale("log")
    axs[0].set_ylabel("log(loss(t) - loss*)", fontsize=20)
    plt.tight_layout()
    plt.savefig("validation_general.pdf")
    plt.show()

    # plt.figure(figsize=(6, 6))
    # plt.title("Generalized solution")
    # for view in [1, 4, 8, 16]:
    #     sgd_loss, optimal = test_generalized("gaussian-0.1", view)
    #     print(optimal)
    #     plt.semilogy(np.abs(sgd_loss - optimal))
    # plt.ylabel("log | loss(t) - loss* |")
    # plt.xlabel("sgd steps (t)")
    # plt.tight_layout()
    # plt.show()
