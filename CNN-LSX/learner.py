import os
from typing import Tuple, Optional, List, Callable
import matplotlib.pyplot as plt
import torch
from captum.attr import InputXGradient
from captum.attr import IntegratedGradients
from captum.attr import LayerGradCam, LayerAttribution
from torch import Tensor, nn, optim
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import global_vars
import utils
from config import SimpleArgumentParser
from critic import Critic
from net import Net, Net2, SimpleConvNet, Net3, Net4
from utils import compute_accuracy
from utils import colored, Loaders
from visualization import ImageHandler
from rtpt import RTPT

Loss = float


class Learner:
    classifier_fn: Callable
    critic_classifier_fn: Callable
    optimizer: Optimizer
    loaders: Optional[Loaders]  # None if the learner is only loaded from checkpoint, and not trained
    optimizer_type: Optional[str]
    test_batch_to_visualize: Optional[Tuple[Tensor, Tensor]]
    model_path: str

    def __init__(self,
                 classifier_fn: Callable,
                 critic_classifier_fn: Callable,
                 loaders: Optional[Loaders],
                 optimizer_type: Optional[str],
                 test_batch_to_visualize: Optional[Tuple[Tensor, Tensor]],
                 model_path: str, explanation_mode: str):
        self.loaders = loaders
        self.optimizer_type = optimizer_type
        self.test_batch_to_visualize = test_batch_to_visualize
        self.model_path = model_path
        self.explanation_mode = explanation_mode
        self.critic_classifier_fn = critic_classifier_fn
        self.critic = None  # dummy
        self.classifier = classifier_fn().to(global_vars.DEVICE)
        self.classifier_fn = classifier_fn

    def load_state(self, path: str):
        path = os.path.join(utils.get_git_root(), path)
        if global_vars.DEVICE == 'cuda':
            checkpoint: dict = torch.load(path)
        elif global_vars.DEVICE == 'cpu':
            checkpoint: dict = torch.load(path, map_location=torch.device('cpu'))
        else:
            raise NotImplementedError(f"dealing with device {global_vars.DEVICE} not implemented")
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.classifier.train()

    def save_state(self, path: str, epoch: int, loss: float):
        if path and global_vars.LOGGING:  # empty model path means we don't save the model
            # first rename the previous model file, as torch.save does not necessarily overwrite the old model.
            if os.path.isfile(path):
                os.replace(path, path + "_previous.pt")

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.classifier.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }, path)
            print(colored(200, 100, 0, f"Saved model to {path}"))

    def pretrain(self,
                 learning_rate: float,
                 learning_rate_step: float,
                 lr_scheduling: bool,
                 n_epochs: int,
                 ) -> Tuple[Loss, Loss]:
        init_loss, end_loss = self.train(learning_rate=learning_rate,
                                         learning_rate_step=learning_rate_step,
                                         n_epochs=n_epochs,
                                         lr_scheduling=lr_scheduling,
                                         classification_loss_weight=1.0,
                                         explanation_loss_weight=0.0,
                                         critic_lr=None)
        self.save_state(f"./{global_vars.LOGGING.writer.log_dir}/pretrained_model.pt", epoch=-1, loss=end_loss)
        return init_loss, end_loss

    def train(self,
              learning_rate: float,
              learning_rate_step: float,
              n_epochs: int,
              lr_scheduling: bool,
              classification_loss_weight: float,
              explanation_loss_weight: float,
              critic_lr: Optional[float]
              ) -> Tuple[Loss, Loss]:
        if n_epochs<=0:
            rtpt = RTPT(name_initials='WS', experiment_name='Learner-Critic', max_iterations=1)
        else:
            rtpt = RTPT(name_initials='WS', experiment_name='Learner-Critic', max_iterations=n_epochs)
        rtpt.start()
        if self.loaders is None or self.optimizer_type is None:
            raise ValueError("Can't train, because the learner is in evaluation mode.")

        self.classifier.train()
        self.initialize_optimizer(learning_rate)
        classification_loss_fn: Module = nn.CrossEntropyLoss(self.loaders.train.dataset.class_weights.float().to(global_vars.DEVICE))
        scheduler = StepLR(self.optimizer, step_size=1, gamma=learning_rate_step)

        start_classification_loss: Optional[Loss] = None
        end_classification_loss: Optional[Loss] = None
        mean_critic_loss: Loss = 0
        for current_epoch in range(n_epochs):
            print(f"epoch {current_epoch}")

            for n_current_batch, (inputs, labels) in enumerate(self.loaders.train):
                self.optimizer.zero_grad()

                inputs, labels = inputs.to(global_vars.DEVICE), labels.to(global_vars.DEVICE)

                outputs = self.classifier(inputs)
                classification_loss = classification_loss_weight * classification_loss_fn(outputs, labels)

                explanation_loss_total_weight = 0.0
                if critic_lr is not None:  # if we are not in pretraining

                    # this will add to the gradients of the learner classifier's weights
                    mean_critic_loss = self.train_critic_on_explanations(critic_lr=critic_lr)

                    # however, as the gradients of the critic loss are added in each critic step,
                    # they are divided by the length of the critic set so the length of the critic set does
                    # not influence the experiments by modulating the number of added gradients.
                    explanation_loss_total_weight = explanation_loss_weight / len(self.loaders.critic)
                    for x in self.classifier.parameters():
                        x.grad *= explanation_loss_total_weight

                # additionally, add the gradients of the classification loss
                classification_loss.backward()

                if n_current_batch == 0:
                    start_classification_loss = classification_loss.item()
                end_classification_loss = classification_loss.item()

                self.optimizer.step()
                self.log_values(classification_loss=classification_loss.item(),
                                pretraining_mode=critic_lr is None,
                                current_epoch=current_epoch,
                                n_current_batch=n_current_batch,
                                n_epochs=n_epochs,
                                mean_critic_loss=mean_critic_loss,
                                explanation_loss_total_weight=explanation_loss_total_weight,
                                finetuning_mode=False)

                rtpt.step(subtitle=f"epoch:{current_epoch+1}/{n_epochs}")

            if global_vars.DEVICE != 'cpu':  # on the cpu I assume it's not a valuable run which needs saving
                self.save_state(self.model_path, epoch=n_epochs, loss=end_classification_loss)
            if lr_scheduling:
                scheduler.step()

        self.terminate_writer()
        return start_classification_loss, end_classification_loss

    def train_critic_on_explanations(self,
                                     critic_lr: float,
                                     explanation_mode: Optional[str] = None):

        if explanation_mode is None:
            explanation_mode = self.explanation_mode
        self.critic = Critic(self.critic_classifier_fn, critic_loader=self.loaders.critic,
                             log_interval_critic=global_vars.LOGGING.critic_log_interval if
                             global_vars.LOGGING else None,
                             shuffle_data=True, class_weights=self.loaders.train.dataset.class_weights.float())
        explanation_batches = [x for [x, _] in self.get_labeled_explanation_batches(self.loaders.critic,
                                                                                    explanation_mode)]
        critic_mean_loss: float
        *_, critic_mean_loss = self.critic.train(explanation_batches, critic_lr)

        return critic_mean_loss

    def finetune_on_explanations(self, args: SimpleArgumentParser):
        self.get_expl_train_data(args)
        print("Explanations for training data loaded")

        rtpt = RTPT(name_initials='WS', experiment_name='Learner-Critic', max_iterations=args.n_finetuning_epochs)
        rtpt.start()
        if self.loaders is None or self.optimizer_type is None:
            raise ValueError("Can't train, because the learner is in evaluation mode.")

        # re-initialaize classifier
        self.classifier = self.classifier_fn().to(global_vars.DEVICE)

        self.classifier.train()
        self.initialize_optimizer(args.learning_rate_finetune)
        classification_loss_fn: Module = nn.CrossEntropyLoss(self.loaders.train.dataset.class_weights.float().to(global_vars.DEVICE))
        expl_loss_fn: Module = nn.MSELoss()
        scheduler = StepLR(self.optimizer, step_size=1, gamma=args.learning_rate_step)

        start_classification_loss: Optional[Loss] = None
        end_classification_loss: Optional[Loss] = None
        for current_epoch in range(args.n_finetuning_epochs):
            print(f"epoch {current_epoch}")

            for n_current_batch, (inputs, expls, labels) in enumerate(self.loaders.train_expl):
                self.optimizer.zero_grad()

                inputs, gt_expls, labels = inputs.to(global_vars.DEVICE), expls.to(global_vars.DEVICE), labels.to(global_vars.DEVICE)

                outputs = self.classifier(inputs)

                expls = self.get_explanation_batch(inputs, labels)

                classification_loss = classification_loss_fn(outputs, labels)

                expl_loss = expl_loss_fn(gt_expls, expls)

                loss = classification_loss + args.explanation_loss_weight_finetune * expl_loss
                loss.backward()

                self.optimizer.step()

                if n_current_batch == 0:
                    start_classification_loss = classification_loss.item()
                end_classification_loss = classification_loss.item()

                self.log_values(classification_loss=classification_loss.item(),
                                pretraining_mode=False,
                                current_epoch=current_epoch,
                                n_current_batch=n_current_batch,
                                n_epochs=args.n_finetuning_epochs,
                                mean_critic_loss=expl_loss,
                                explanation_loss_total_weight=args.explanation_loss_weight_finetune,
                                finetuning_mode=True)

                rtpt.step(subtitle=f"epoch:{current_epoch+1}/{args.n_finetuning_epochs}")

            if global_vars.DEVICE != 'cpu':  # on the cpu I assume it's not a valuable run which needs saving
                self.save_state(f"./{global_vars.LOGGING.writer.log_dir}/finetuned_model.pt",
                                epoch=args.n_epochs+args.n_finetuning_epochs,
                                loss=end_classification_loss)
            if args.lr_scheduling:
                scheduler.step()

        self.terminate_writer()
        return start_classification_loss, end_classification_loss

    def get_expl_train_data(self, args):
        data = self.get_detached_labeled_explanation_input_batches(self.loaders.train)
        dataset_train = torch.utils.data.dataset.TensorDataset(torch.stack(data[0], dim=0),
                                                               torch.stack(data[1], dim=0),
                                                               torch.stack(data[2], dim=0))
        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size, num_workers=0, shuffle=True
        )
        self.loaders.train_expl = train_loader

    def get_labeled_explanation_batches(self,
                                        dataloader: DataLoader,
                                        explanation_mode: Optional[str] = None) -> List[List[Tensor]]:
        labeled_explanation_batches = []
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(global_vars.DEVICE), labels.to(global_vars.DEVICE)
            labeled_explanation_batches.append([self.get_explanation_batch(inputs, labels, explanation_mode), labels])
        return labeled_explanation_batches

    def get_detached_labeled_explanation_input_batches(self,
                                        dataloader: DataLoader,
                                        explanation_mode: Optional[str] = None) -> List[List[Tensor]]:
        input_all = []
        labels_all = []
        explanation_all = []
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(global_vars.DEVICE), labels.to(global_vars.DEVICE)
            explanation_all += self.get_explanation_batch(inputs, labels, explanation_mode).detach().cpu()
            input_all += inputs.detach().cpu()
            labels_all += labels.detach().cpu()
        return input_all, explanation_all, labels_all

    def log_values(self, classification_loss: float, pretraining_mode: bool, finetuning_mode: bool, current_epoch: int,
                   n_current_batch: int, n_epochs: int, mean_critic_loss: float, explanation_loss_total_weight: float):
        if global_vars.LOGGING:
            if n_current_batch % global_vars.LOGGING.log_interval == 0:
                self.log_training_details(explanation_loss_total_weight=explanation_loss_total_weight,
                                          mean_critic_loss=mean_critic_loss,
                                          classification_loss=classification_loss,
                                          learning_rate=self.optimizer.param_groups[0]['lr'],
                                          finetuning_mode=finetuning_mode)
            if n_current_batch % global_vars.LOGGING.log_interval_accuracy == 0 and self.loaders.test:
                self.log_accuracy(finetuning_mode)
                ImageHandler.add_gradient_images(self.test_batch_to_visualize, self, "2: during training",
                                                 global_step=global_vars.global_step)
            if pretraining_mode or finetuning_mode:
            #     print(f'{colored(100, 50, 100, "pretraining:")}')
                global_vars.global_step += 1
                # in pretraining mode the global step is not increased in the critic, so it needs to be done here.

            progress_percentage: float = 100 * current_epoch / n_epochs
            print(f'{colored(0, 150, 100, str(global_vars.LOGGING.run_name))}: '
                  f'epoch {current_epoch}, '
                  f'learner batch {n_current_batch} of {n_epochs} epochs '
                  f'({colored(200, 200, 100, f"{progress_percentage:.0f}%")})]')

    def train_from_args(self, args: SimpleArgumentParser):
        init_loss, end_loss = self.train(learning_rate=args.learning_rate,
                          learning_rate_step=args.learning_rate_step,
                          n_epochs=args.n_epochs,
                          lr_scheduling=args.lr_scheduling,
                          classification_loss_weight=args.classification_loss_weight,
                          explanation_loss_weight=args.explanation_loss_weight,
                          critic_lr=args.learning_rate_critic)

        self.save_state(f"./{global_vars.LOGGING.writer.log_dir}/joint_model.pt", epoch=args.n_epochs,
                        loss=end_loss)
        return init_loss, end_loss

    def pretrain_from_args(self, args: SimpleArgumentParser):
        return self.pretrain(args.pretrain_learning_rate, args.learning_rate_step, args.lr_scheduling,
                             args.n_pretraining_epochs)

    @staticmethod
    def log_training_details(explanation_loss_total_weight, mean_critic_loss, classification_loss,
                             learning_rate, finetuning_mode):

        log_str = 'Learner_Training'
        if finetuning_mode:
            log_str = 'Finetuning'

        # add scalars to writer
        global_step = global_vars.global_step
        if global_vars.LOGGING:
            if explanation_loss_total_weight:
                total_loss = mean_critic_loss * explanation_loss_total_weight + classification_loss
            else:
                total_loss = classification_loss
            global_vars.LOGGING.writer.add_scalar(f"{log_str}/Explanation", mean_critic_loss,
                                                  global_step=global_step)
            global_vars.LOGGING.writer.add_scalar(f"{log_str}/Classification", classification_loss,
                                                  global_step=global_step)
            global_vars.LOGGING.writer.add_scalar(f"{log_str}/Total", total_loss,
                                                  global_step=global_step)
            global_vars.LOGGING.writer.add_scalar(f"{log_str}/Learning_Rate", learning_rate,
                                                  global_step=global_step)

            # print statistics
            print(f'Loss: {total_loss:.3f} ='
                  f' {classification_loss:.3f}(classification) + {explanation_loss_total_weight}(lambda)'
                  f'*{mean_critic_loss:.3f}(explanation)')

    @staticmethod
    def terminate_writer():
        if global_vars.LOGGING:
            global_vars.LOGGING.writer.flush()
            global_vars.LOGGING.writer.close()

    def integrated_gradient(self, input_images: Tensor, labels: Tensor) -> Tensor:
        integrated_gradients = IntegratedGradients(self.classifier.forward)
        input_images.requires_grad = True
        int_grad: Tensor = integrated_gradients.attribute(inputs=input_images, target=labels)
        int_grad = int_grad.float()
        return int_grad

    def input_gradient(self, input_images: Tensor, labels: Tensor) -> Tensor:
        input_x_gradient = InputXGradient(self.classifier.forward)
        input_images.requires_grad = True
        gradient_x_input: Tensor = input_x_gradient.attribute(inputs=input_images, target=labels)
        # gradient: Tensor = gradient_x_input / input_images
        return gradient_x_input

    def gradcam(self, input_images: Tensor, labels: Tensor) -> Tensor:
        last_conv_layer = utils.get_last_conv_layer(self.classifier)
        gradcam = LayerGradCam(self.classifier, last_conv_layer)
        input_images.requires_grad = True
        gradcam: Tensor = gradcam.attribute(inputs=input_images, target=labels, relu_attributions=False)

        # expl = F.relu(gradcam)

        norm_saliencies = utils.norm_saliencies_fast(gradcam)

        # up = nn.UpsamplingBilinear2d(size=(28, 28))
        return LayerAttribution.interpolate(norm_saliencies, (28, 28))


    @staticmethod
    def clip_and_rescale(images: Tensor) -> Tensor:
        # if not self.disable_gradient_clipping:
        # clip negative gradients to zero (don't distinguish between "no impact" and "negative impact" on the label)
        images[images < 0] = 0
        return ImageHandler.rescale_to_zero_one(images)

    def get_explanation_batch(self, inputs: Tensor, labels: Tensor, explanation_mode: Optional[str] = None, save=False) -> Tensor:
        if explanation_mode is None:
            explanation_mode = self.explanation_mode

        if explanation_mode == "input_x_gradient":
            input_gradient = self.input_gradient(inputs, labels)
            clipped_rescaled_input_gradient = self.clip_and_rescale(input_gradient)
            return clipped_rescaled_input_gradient
            # if explanation_mode == "input_x_gradient":
            #     return clipped_rescaled_input_gradient * inputs
            # else:
            #     return clipped_rescaled_input_gradient
        elif explanation_mode == "integrated_gradient" or explanation_mode == "input_x_integrated_gradient":
            integrated_gradient = self.integrated_gradient(inputs, labels)
            clipped_rescaled_integrated_gradient = self.clip_and_rescale(integrated_gradient)
            if self.explanation_mode == "input_x_integrated_gradient":
                return clipped_rescaled_integrated_gradient * inputs
            else:
                return clipped_rescaled_integrated_gradient
        elif explanation_mode == "input":
            return inputs
        elif explanation_mode == 'gradcam':
            gradcam = self.gradcam(inputs, labels)
            return gradcam
        else:
            raise NotImplementedError(f"unknown explanation mode '{explanation_mode}'")

    def predict(self, images: Tensor) -> Tensor:
        outputs = self.classifier(images)
        _, prediction = torch.max(outputs, 1)
        return prediction

    def log_accuracy(self, finetuning_mode: bool):
        global_step = global_vars.global_step
        training_accuracy = compute_accuracy(self.classifier, self.loaders.train, global_vars.LOGGING.n_test_batches)
        test_accuracy = compute_accuracy(self.classifier, self.loaders.test)

        if not self.critic:
            self.critic = Critic(self.critic_classifier_fn, critic_loader=self.loaders.critic,
                                 log_interval_critic=global_vars.LOGGING.critic_log_interval if
                                 global_vars.LOGGING else None,
                                 shuffle_data=False, class_weights=self.loaders.train.dataset.class_weights.float())

        critic_training_accuracy = compute_accuracy(classifier=self.critic.classifier,
                                                    data=self.get_labeled_explanation_batches(self.loaders.critic),
                                                    n_batches=len(self.loaders.test)
                                                    )
        critic_test_accuracy_input = compute_accuracy(classifier=self.critic.classifier,
                                                      data=self.loaders.test,
                                                      )
        critic_training_accuracy_input = compute_accuracy(classifier=self.critic.classifier,
                                                          data=self.loaders.critic,
                                                          n_batches=len(self.loaders.test)
                                                          )

        print(f'accuracy training: {training_accuracy:3f}, accuracy testing: {test_accuracy:.3f}, '
                                 f'accuracy critic training:{critic_training_accuracy:3f}')

        log_str = 'Learner_Training'
        if finetuning_mode:
            log_str = 'Finetuning'

        if global_vars.LOGGING:
            global_vars.LOGGING.writer.add_scalar(f"{log_str}/Training_Accuracy", training_accuracy,
                                                  global_step=global_step)
            global_vars.LOGGING.writer.add_scalar(f"{log_str}/Test_Accuracy", test_accuracy,
                                                  global_step=global_step)
            global_vars.LOGGING.writer.add_scalar("Critic_Training/Training_Accuracy", critic_training_accuracy,
                                                  global_step=global_step)
            global_vars.LOGGING.writer.add_scalar("Critic_Training/Input_Test_Accuracy", critic_test_accuracy_input,
                                                  global_step=global_step)
            global_vars.LOGGING.writer.add_scalar("Critic_Training/Input_Training_Accuracy",
                                                  critic_training_accuracy_input,
                                                  global_step=global_step)

    def initialize_optimizer(self, learning_rate):
        if self.optimizer_type == "adadelta":
            self.optimizer = optim.Adadelta(self.classifier.parameters(), lr=learning_rate)
        elif self.optimizer_type == "adam":
            self.optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"optimizer '{self.optimizer_type}' invalid")

    def get_labeled_explanations(self, test_loader: DataLoader, mode: str) -> Tuple[Tensor, Tensor]:
        """get all explanations together with the labels, and don't combine them into batches."""
        explanations = []
        explanation_labels = []
        for inputs, labels in test_loader:
            explanation_batch: List[Tensor] = list(self.get_explanation_batch(inputs, labels, mode))
            # labeled_explanation_batch: List[Tuple[Tensor, int]] = list(zip(explanation_batch, list(labels)))
            explanations.extend(explanation_batch)
            explanation_labels.extend(labels)
        explanation_tensor = torch.stack(explanations)
        label_tensor = torch.stack(explanation_labels)
        return explanation_tensor, label_tensor
