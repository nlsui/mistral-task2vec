# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import itertools
import logging
import math
import random
from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from utils import get_device


class MistralEmbedding:
    def __init__(self, hessian, scale, meta=None):
        self.hessian = np.array(hessian)
        self.scale = np.array(scale)
        self.meta = meta


class MistralProbeNetwork(ABC, nn.Module):
    """Abstract class that all probe networks should inherit from.

    This is a standard torch.nn.Module but needs to expose a classifier property that returns the final classicifation
    module (e.g., the last fully connected layer).
    """


class MistralTask2Vec:

    def __init__(self, model: MistralProbeNetwork, skip_layers=0, max_samples=None,
                 method_opts=None, loader_opts=None, bernoulli=False, seed=0):
        if method_opts is None:
            method_opts = {}
        if loader_opts is None:
            loader_opts = {}
        assert skip_layers >= 0

        self.model = model
        # Fix batch norm running statistics (i.e., put batch_norm layers in eval mode)
        self.model.train()
        self.device = get_device(self.model)
        self.skip_layers = skip_layers
        self.max_samples = max_samples
        self.method_opts = method_opts
        self.loader_opts = loader_opts
        self.bernoulli = bernoulli
        self.seed = seed
        self.loss_fn = nn.CrossEntropyLoss() if not self.bernoulli else nn.BCEWithLogitsLoss()
        self.loss_fn = self.loss_fn.to(self.device)

    def embed(self, dataset: Dataset):
        self.compute_montecarlo_fisher(dataset, **self.method_opts)
        embedding = self.extract_embedding(self.model)
        return embedding

    def compute_montecarlo_fisher(self, dataset: Dataset, epochs: int = 1, max_samples=None, loader_opts: dict = None):
        logging.info("Using Monte Carlo Fisher Information Calculation")

        # Seed setting for deterministic results
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Prepare data loader
        if loader_opts is None:
            loader_opts = {}

        data_loader = _get_loader(dataset, **self.loader_opts)

        if max_samples is not None:
            n_batches = min(math.floor(max_samples / data_loader.batch_size) - 1, len(data_loader))
        else:
            n_batches = len(data_loader)

        device = get_device(self.model)

        # Initialize gradient accumulators for Fisher Information calculation
        for p in self.model.parameters():
            p.grad2_acc = torch.zeros_like(p.data)
            p.grad_counter = 0

        logging.info("Computing Fisher Information...")

        for k in range(epochs):
            logging.info(f"\tEpoch {k + 1}/{epochs}")

            for i, (data, target) in tqdm(enumerate(itertools.islice(data_loader, 0, n_batches)), total=n_batches,
                                          leave=False, desc="Computing Fisher"):
                data = data.to(device)

                # Forward pass through the model
                output = self.model(data)

                # Access the logits from the model output
                logits = output.logits

                # Reshape logits to [batch_size * sequence_length, num_classes]
                logits = logits.view(-1, logits.size(-1))

                # Sample the target based on logits
                if self.bernoulli:
                    target = torch.bernoulli(F.sigmoid(logits)).detach()
                else:
                    # Sample a token from each distribution (across the num_classes dimension)
                    target = torch.multinomial(F.softmax(logits, dim=-1), 1).detach().view(-1)

                # Calculate loss and backpropagate
                loss = self.loss_fn(logits, target)
                self.model.zero_grad()
                loss.backward()

                # Accumulate Fisher Information gradient (squared gradients)
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad2_acc += p.grad.data ** 2
                        p.grad_counter += 1

        # Normalize gradients
        for p in self.model.parameters():
            if p.grad_counter == 0:
                del p.grad2_acc
            else:
                p.grad2_acc /= p.grad_counter

        logging.info("Fisher Information calculation done")

    def extract_embedding(self, model: MistralProbeNetwork):
        """
        Reads the values stored by `compute_fisher` and returns them in a common format that describes the diagonal of the
        Fisher Information Matrix for each layer.

        :param model:
        :return:
        """
        hess, scale = [], []
        for name, module in model.named_modules():
            if module is model.lm_head:
                continue
            # The variational Fisher approximation estimates the variance of noise that can be added to the weights
            # without increasing the error more than a threshold. The inverse of this is proportional to an
            # approximation of the hessian in the local minimum.
            if hasattr(module, 'logvar0') and hasattr(module, 'loglambda2'):
                logvar = module.logvar0.view(-1).detach().cpu().numpy()
                hess.append(np.exp(-logvar))
                loglambda2 = module.loglambda2.detach().cpu().numpy()
                scale.append(np.exp(-loglambda2).repeat(logvar.size))
            # The other Fisher approximation methods directly approximate the hessian at the minimum
            elif hasattr(module, 'weight') and hasattr(module.weight, 'grad2_acc'):
                grad2 = module.weight.grad2_acc.cpu().detach().numpy()
                filterwise_hess = grad2.reshape(grad2.shape[0], -1).mean(axis=1)
                hess.append(filterwise_hess)
                scale.append(np.ones_like(filterwise_hess))
        return MistralEmbedding(hessian=np.concatenate(hess), scale=np.concatenate(scale), meta=None)


def _get_loader(trainset, testset=None, batch_size=1, num_workers=2, num_samples=None, drop_last=True):
    # Since we are dealing with sequences, we don't need to calculate class counts or weights
    sampler = None

    # Handle multi-threaded loading for larger datasets
    num_workers = num_workers if not isinstance(trainset, torch.utils.data.TensorDataset) else 0

    # Create the DataLoader for the training set
    train_loader = torch.utils.data.DataLoader(
        trainset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        shuffle=True  # Shuffle sequences to randomize batches
    )

    if testset is None:
        return train_loader
    else:
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=num_workers
        )
        return train_loader, test_loader
