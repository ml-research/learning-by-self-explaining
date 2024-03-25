from tap import Tap


def _colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[0m".format(r, g, b, text)


class SimpleArgumentParser(Tap):
    training_mode: str = "pretrain_and_joint"
    logging_disabled: bool = False
    lr_scheduling: bool = False
    random_seed: int = 30
    dataset: str = 'mnist'
    model: str = 'Net1'
    model_pt: str = None
    vanilla_model_pt: str = None
    no_cuda: bool = False

    # Training Details
    batch_size: int = 128
    batch_size_critic: int = 128
    test_batch_size: int = 128
    learning_rate: float = 0.01
    learning_rate_finetune: float = 0.001
    learning_rate_step: float = 0.7
    learning_rate_critic: float = 0.2
    pretrain_learning_rate: float = 0.05
    classification_loss_weight: float = 50  # high by default,
    explanation_loss_weight: float = 50  # high by default,
    explanation_loss_weight_finetune: float = 50  # high by default,
    # as the critic loss has a longer way to the weights, and therefore less influence.
    optimizer: str = 'adadelta'
    explanation_mode: str = 'input_x_gradient'

    # Dataset sizes
    # n_training_batches: int = 400  # 400
    n_critic_batches: int = 68  # these are taken from the training set
    sep_critic_set: bool = False # if True the critic set is a distinct set from the training set
    # n_test_batches: int = 5
    n_epochs: int = 40
    n_pretraining_epochs: int = 10
    n_finetuning_epochs: int = 50
    # disable_critic_shuffling: bool = False
    few_shot_train_percent: float = 1.0 # percentage of training samples from the original training set

    log_interval: int = 1
    # in case some day learner values seem too much (e.g. if Tensorboard is overburdened and slow).
    # Setting this to a different value than 1 will lead to the critic plot having somewhat confusing holes.
    log_interval_critic: int = 5
    log_interval_pretraining: int = log_interval
    # Setting this to a different value than log_interval will lead to pre- and joint training having
    # different logging intervals.
    log_interval_accuracy: int = 50
    # setting this to a lower value will reduce performance significantly.

    rand_critic: bool = False # if True the critic set is a distinct set from the training set

    render_enabled: bool = False

    run_name: str = ""

    # def process_args(self):
    #     low_number_of_iterations = 50
    #     # n_iterations = self.n_iterations
    #     if not self.logging_disabled and n_iterations < low_number_of_iterations:
    #         # if we have so few iterations then it's probably a debug run.
    #         print(_colored(200, 150, 0, f"Logging everything, as there are only {n_iterations} iterations"))
    #         self.log_interval = 1
    #         self.log_interval_critic = 1
    #         self.log_interval_pretraining = 1
    #         self.log_interval_accuracy = 1

    @property
    def joint_iterations(self) -> int:
        return self.n_epochs * self.n_training_batches * self.n_critic_batches

    @property
    def pretraining_iterations(self) -> int:
        return self.n_pretraining_epochs * self.n_training_batches

    @property
    def finetuning_iterations(self) -> int:
        return self.n_finetuning_epochs * self.n_training_batches

    @property
    def n_iterations(self) -> int:
        if self.training_mode == 'joint' or self.training_mode == "pretrained":
            return self.joint_iterations
        elif self.training_mode == 'pretrain_and_joint':
            return self.pretraining_iterations + self.joint_iterations
        elif self.training_mode == 'pretrain_and_joint_and_finetuning':
            return self.pretraining_iterations + self.joint_iterations + self.finetuning_iterations
        elif self.training_mode == 'finetuning':
            return self.finetuning_iterations
        elif self.training_mode == 'only_critic':
            return self.n_critic_batches  # critic only trains one episode
        elif self.training_mode == 'only_classification':
            return self.pretraining_iterations
        elif self.training_mode == 'one_critic_pass':
            return self.pretraining_iterations + self.n_critic_batches
        else:
            raise ValueError(f"invalid training mode: {self.training_mode}")
