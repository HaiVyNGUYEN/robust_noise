# Training More Robust Classification Model via Discriminative Loss and Gaussian Noise Injection

This Repo contains all the files related to the paper.

![Propose method](https://github.com/HaiVyNGUYEN/robust_noise/blob/master/illustrative/noisy_training.png)

![...](https://github.com/HaiVyNGUYEN/robust_noise/blob/master/illustrative/tsne.png)

## Dependencies

The code is implemented based mainly on python library Pytorch (and torchvision). All needed libraries can be found in  [requirements.txt](https://github.com/HaiVyNGUYEN/ld_official/blob/master/requirements.txt). The code is supposed to be run in Linux but can be easily adapted for other systems. We strongly recommend to create virtual environment for a proper running (such as conda virtual env). This can be easily done in linux terminal as follow:
```
conda create -n yourenvname python=x.x anaconda
```
Then, to activate this virtual env:
```
conda activate yourenvname
```
To install a package in this virtual env:
```
conda install -n yourenvname [package]
```

To quit this env:

```
conda deactivate
```

## Data

In this work, we use publicly available datasets [SVHN](http://ufldl.stanford.edu/housenumbers/), [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html). Besides, we also use our own dataset as described in the paper.

## Running for experiments

The process is totally the same for different datasets. Hence, we put here only the scheme for CIFAR10, but the method is easily applied for any dataset. We intentionally make separate files in short format, avoiding inter-connection pipelines between files, so that readers can easily understand and adapt for their own purposes. Besides, we also give example of how to apply the perturbations on the data using pytorch in [perturbations](https://github.com/HaiVyNGUYEN/robust_noise/tree/master/perturbations).

