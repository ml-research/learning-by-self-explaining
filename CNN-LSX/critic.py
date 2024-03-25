import random
from statistics import mean
from typing import Any, List, Tuple, Optional, Callable

from torch import Tensor, nn, optim
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader

import global_vars
from net import Net, Net2, SimpleConvNet, Net3, Net4

Loss = float


class Critic:

    def __init__(self, classifier_fn: Callable, critic_loader: DataLoader[Any],
                 log_interval_critic: Optional[int],
                 shuffle_data: bool, class_weights, random=False
                 ):
        self.classifier_fn = classifier_fn
        self.classifier = self.classifier_fn().to(global_vars.DEVICE)
        self.critic_loader = critic_loader
        self.log_interval_critic = log_interval_critic
        self.shuffle_data: bool = shuffle_data
        self.class_weights = class_weights
        self.random = random


    def train(self, explanations: List[Tensor], critic_learning_rate: float) -> Tuple[float, float, float]:

        self.classifier.train()
        # shuffle the data before each critic training.
        if explanations:
            zipped = list(zip(explanations, list(self.critic_loader)))
            if self.shuffle_data:
                shuffled_zipped = random.sample(zipped, len(zipped))
            else:
                shuffled_zipped = zipped

            permuted_explanations, permuted_critic_set = zip(*shuffled_zipped)
        else:
            permuted_explanations = []  # if explanations are empty, then permuted explanations are also empty.
            permuted_critic_set = list(self.critic_loader)
            random.shuffle(permuted_critic_set)

        critic_loss: Module = nn.CrossEntropyLoss(self.class_weights.to(global_vars.DEVICE))
        optimizer: Optimizer = optim.Adadelta(self.classifier.parameters(), lr=critic_learning_rate)

        losses: List[float] = []

        for n_current_batch, (inputs, labels) in enumerate(permuted_critic_set):
            inputs, labels = inputs.to(global_vars.DEVICE), labels.to(global_vars.DEVICE)

            losses.append(self._process_batch(critic_loss, permuted_explanations,
                                              inputs, labels, n_current_batch,
                                              optimizer))
            global_vars.global_step += 1

        return losses[0], losses[-1], mean(losses)

    def _process_batch(self, loss_function: nn.Module, explanations: List[Tensor], inputs: Tensor, labels: Tensor,
                       n_current_batch: int, optimizer) -> Loss:

        optimizer.zero_grad()

        explanation_batch: Tensor
        if explanations:
            explanation_batch = explanations[n_current_batch]
        else:  # if trained without explanation, just train on the input.
            explanation_batch = inputs
        outputs = self.classifier(explanation_batch)
        # if random critic, permutate gt labels randomly
        if self.random:
            print('Using random critic!')
            idx = torch.randperm(labels.shape[0])
            labels = labels[idx]
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        self._log_results(loss, n_current_batch)
        return loss.item()

    def _log_results(self, loss, n_current_batch):
        if self.log_interval_critic and n_current_batch % self.log_interval_critic == 0:
            # if n_current_batch == 0 or n_current_batch == self.cfg.n_critic_batches - 1:

            print(f'crit_batch = {n_current_batch}, loss.item() = {loss.item():.3f}')
            self.add_scalars_to_writer(loss)

    def add_scalars_to_writer(self, loss):
        if global_vars.LOGGING:
            global_vars.LOGGING.writer.add_scalar("Critic_Training/Critic_Loss",
                                                  loss.item(),
                                                  global_step=global_vars.global_step)
