# Training Models with different methods on CIFAR-10

To make it easier for readers to follow, we intentionally make separate files in short format, avoiding inter-connected pipelines between files, so that readers can easily understand and adapt for their own purposes.

- [download_data.py](https://github.com/HaiVyNGUYEN/robust_noise/blob/master/cifar10/download_data.py): downloading data.
- [double_dataset.py](https://github.com/HaiVyNGUYEN/robust_noise/blob/master/cifar10/double_dataset.py): customized dataset, returning both clean and noisy data (for different methods of training).
- [training_routines.py](https://github.com/HaiVyNGUYEN/robust_noise/blob/master/cifar10/training_routines.py): resnet18 architecture (in pytorch).
- [discriminative_loss_function.py](https://github.com/HaiVyNGUYEN/robust_noise/blob/master/cifar10/discriminative_loss_function.py): our loss function.
- [training_routines.py](https://github.com/HaiVyNGUYEN/robust_noise/blob/master/cifar10/training_routines.py): different training routines for different methods.
- [normal_training.py](https://github.com/HaiVyNGUYEN/robust_noise/blob/master/cifar10/normal_training.py): normal training using standard softmax loss, with standard data augmentation.
- [noisy_training.py](https://github.com/HaiVyNGUYEN/robust_noise/blob/master/cifar10/noisy_training.py): training using standard softmax loss, with only noisy data, that is data injected with Gaussian noise on top of standard data augmentation.
- [both_training.py](https://github.com/HaiVyNGUYEN/robust_noise/blob/master/cifar10/both_training.py): training using standard softmax loss, with both clean and noisy data.
- [stability_training.py](https://github.com/HaiVyNGUYEN/robust_noise/blob/master/cifar10/stability_training.py): stability training.
- [our_training.py](https://github.com/HaiVyNGUYEN/robust_noise/blob/master/cifar10/our_training.py): our method.

### Evaluation functions

- [evaluation_methods.py](https://github.com/HaiVyNGUYEN/robust_noise/blob/master/cifar10/evaluation_methods.py): estimating curvature; counting number of correct predictions over K independent noise injections for each test example.




