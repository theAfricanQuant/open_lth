# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from datasets import base, cifar10, mnist, imagenet
from foundations.hparams import DatasetHparams
from platforms.platform import get_platform

registered_datasets = {'cifar10': cifar10, 'mnist': mnist, 'imagenet': imagenet}


def get(dataset_hparams: DatasetHparams, train: bool = True):
    """Get the train or test set corresponding to the hyperparameters."""

    seed = dataset_hparams.transformation_seed or 0

    if dataset_hparams.dataset_name not in registered_datasets:
        raise ValueError(f'No such dataset: {dataset_hparams.dataset_name}')

    use_augmentation = train and not dataset_hparams.do_not_augment
    dataset = (
        registered_datasets[
            dataset_hparams.dataset_name
        ].Dataset.get_train_set(use_augmentation)
        if train
        else registered_datasets[
            dataset_hparams.dataset_name
        ].Dataset.get_test_set()
    )
    # Transform the dataset.
    if train and dataset_hparams.random_labels_fraction is not None:
        dataset.randomize_labels(seed=seed, fraction=dataset_hparams.random_labels_fraction)

    if train and dataset_hparams.subsample_fraction is not None:
        dataset.subsample(seed=seed, fraction=dataset_hparams.subsample_fraction)

    if train and dataset_hparams.blur_factor is not None:
        if isinstance(dataset, base.ImageDataset):
            dataset.blur(seed=seed, blur_factor=dataset_hparams.blur_factor)

        else:
            raise ValueError('Can blur images.')
    if dataset_hparams.unsupervised_labels is not None:
        if dataset_hparams.unsupervised_labels != 'rotation':
            raise ValueError(
                f'Unknown unsupervised labels: {dataset_hparams.unsupervised_labels}'
            )
        elif not isinstance(dataset, base.ImageDataset):
            raise ValueError('Can only do unsupervised rotation to images.')
        else:
            dataset.unsupervised_rotation(seed=seed)

    # Create the loader.
    return registered_datasets[dataset_hparams.dataset_name].DataLoader(
        dataset, batch_size=dataset_hparams.batch_size, num_workers=get_platform().num_workers)


def iterations_per_epoch(dataset_hparams: DatasetHparams):
    """Get the number of iterations per training epoch."""

    if dataset_hparams.dataset_name in registered_datasets:
        num_train_examples = registered_datasets[dataset_hparams.dataset_name].Dataset.num_train_examples()
    else:
        raise ValueError(f'No such dataset: {dataset_hparams.dataset_name}')

    if dataset_hparams.subsample_fraction is not None:
        num_train_examples *= dataset_hparams.subsample_fraction

    return np.ceil(num_train_examples / dataset_hparams.batch_size).astype(int)


def num_classes(dataset_hparams: DatasetHparams):
    """Get the number of classes."""

    if dataset_hparams.dataset_name in registered_datasets:
        num_classes = registered_datasets[dataset_hparams.dataset_name].Dataset.num_classes()
    else:
        raise ValueError(f'No such dataset: {dataset_hparams.dataset_name}')

    if dataset_hparams.unsupervised_labels is not None:
        if dataset_hparams.unsupervised_labels != 'rotation':
            raise ValueError(
                f'Unknown unsupervised labels: {dataset_hparams.unsupervised_labels}'
            )
        else:
            return 4

    return num_classes
