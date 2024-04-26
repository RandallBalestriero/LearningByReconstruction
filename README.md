# Learning by Reconstruction Produces Uninformative Features For Perception
## *Randall Balestriero, Yann LeCun*, [arXiv](https://arxiv.org/abs/2402.11337), [Twitter](https://twitter.com/randall_balestr/status/)

Input space reconstruction is an attractive rep-
resentation learning paradigm. Despite inter-
pretability of the reconstruction and generation,
we identify a misalignment between learning by
reconstruction, and learning for perception. We
show that the former allocates a model’s capac-
ity towards a subspace of the data explaining the
observed variance–a subspace with uninformative
features for the latter. For example, the super-
vised TinyImagenet task with images projected
onto the top subspace explaining 90% of the pixel
variance can be solved with 45% test accuracy.
Using the bottom subspace instead, accounting
for only 20% of the pixel variance, reaches 55%
test accuracy. The features for perception being
learned last explains the need for long training
time, e.g., with Masked Autoencoders. Learning
by denoising is a popular strategy to alleviate that
misalignment. We prove that while some noise
strategies such as masking are indeed beneficial,
others such as additive Gaussian noise are not.
Yet, even in the case of masking, we find that the
benefits vary as a function of the mask’s shape,
ratio, and the considered dataset. While tuning
the noise strategy without knowledge of the per-
ception task seems challenging, we provide first
clues on how to detect if a noise strategy is never
beneficial regardless of the perception task.


*This repository provides code to reproduce the results in the paper (the code is still experimental and being worked on, feel free to raise any comment/issue). We require the usual dependencies that come with `PyTorch`, `Numpy` and the likes. Nothing else is needed.*

## Theoretical alignment between reconstruction and perception with and without denoising task

To reproduce those figures, use the code [alignment.py](./alignment.py) code. It will produce both alignment figures with and without the denoising task. The only thing that the user needs to implement before using it is the dataset loading around line 50:
```
# LOAD YOUR DATASET HERE as numpy arrays with format (N, H, W, C) for X
# and integer classes for y
# X, y = ....
```
everything works out of the box. Internally, the code will use the solver located in the [utils.py](utils.py) file.
By default the code will use the following datasets (which can be modified without issue by the user)
- `ArabicCharacters`
- `ArabicDigits`
- `SVHN`
- `FashionMNIST`
- `CIFAR10`
- `CIFAR100`

and the code can be run with a single argument specifying where to dump the figures as in `python alignment.py --path PATH`

## PCA figures (bottom and top part)

To reproduce those figures, we provide [pca_figure.py](pca_figure.py) which internally uses a plotting utility from the [utils.py](utils.py) file but it otherwise self-contained. It will load a dataset of image, perform eigendecomposition, figure out the split for 25% variance (this can be modified by the user on line 68 and 69) and plot the projected images. The only thing to do for the user is to add their dataset around line 50. We recommend using the `Imagenette` dataset which is `Imagenet` with only 10 classes to be more memory-friendly. Alternatively, one could create the `Imagenet` dataset and manually subset it with `torch.utils.data.Subset`:
```
# LOAD YOUR DATASET AND USE THE FOLLOWING TRANSFORM
transform = v2.Compose(
    [
        v2.PILToTensor(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
    ]
)
# dataset = ...
```

## Supervised Performance on PCA subspaces

To reproduce the figures demonstrating the supervised performance of models on images projected on different PCA subspaces, we provide the [classification_pca.py](classification_pca.py) file. In works out of the box after adding a few lines to accomodate for one's dataloading and training pipeline.


Add you own dataset loading pipeline that should return a numpy array at 
- line 128:
    ```
    # Load X only
    # X = ....
    ```
- line 142:
    ```
    # Load X, y and Xtest, ytest as numpy arrays in RGB format with (N, H, W, C)
    # X, y = ....
    # Xtest, ytest = ...
    ```
- line 146:
    ```
    # Load X, y and Xtest, ytest as numpy arrays in RGB format with (N, H, W, C)
    # X, y = ....
    # Xtest, ytest = ...
    ```
and launch your favorite supervised training pipeline in line 236 with parameters:
```
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
```
in our case we leverage PytorchLIghtning with the following settings:
```
class MyLightningModule(pl.Module):
    def create_modules(self):
        if self.arch == "MLP":
            self.model = MLP(self.input_shape, self.outputs, depth=self.depth)
        elif self.arch == "R9":
            self.model = ResNet9(self.input_shape, self.outputs)

    def create_metrics(self):
        self.train_error = torchmetrics.classification.MulticlassAccuracy(
            self.outputs[0]
        )
        self.val_metric = torchmetrics.classification.MulticlassAccuracy(
            self.outputs[0]
        )

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch, batch_idx):
        x, y = batch
        yhat = self.model(x)
        if not self.training:
            self.log("eval_accuracy", self.val_metric(yhat.argmax(1), y))
        return nn.functional.mse_loss(
            yhat, torch.nn.functional.one_hot(y, self.outputs[0]).float()
        )
```

## Supervised Guidance 

To reproduce that experiment that shows how a given reconstruction error can be achieved by both an autoencoder with good perception performance and one with poor performance, simply take your favorite autoencoder training code and add a weighted (with a weight dneoted as `guidance`) supervised loss to it:
```
x, y = batch
h = encoder(x)
xrec = decoder(h)
rec_loss = torch.nn.functional.mse_loss(xrec, x)
yhat = self.probe(h)
sup_loss = guidance * torch.nn.functional.cross_entropy(yhat, y)
loss = rec_loss + sup_loss
```
where the weight `guidance` needs to be small enough to not impact the reconstruction performance.

## Autoencoder Architectures

All the architectures used in the paper are provided within the [utils.py](utils.py) file:
- `DCAE`
- `MLPAE`
- `ResNetAutoEncoder`

## Testing (empirical validation of theorems)

We provide in the [tests.py](tests.py) file the tests that empirically validate the theoretical claims of the paper. The code will automatically generate figures comparing the theoretical optimal solutions found in our theorems to solutions found using gradient descent. We produced results will reach a different of near 0 (up to round off error), validating the closed-form solution we obtained.