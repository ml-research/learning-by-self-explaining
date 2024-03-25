import json
import os
import random
from datetime import datetime
from typing import Tuple, Any, Optional, List, Union

import git
import numpy as np
import torch.cuda
import torch.multiprocessing
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO, Evaluator
from medmnist.dataset import MedMNIST2D, DermaMNIST, PneumoniaMNIST, ChestMNIST
from PIL import Image

import global_vars
import load_decoy_mnist as load_decoy_mnist
from config import SimpleArgumentParser
from helper_types import Loaders, Logging
from net import MLP, Net, Net2, SimpleConvNet, Net3, Net4, CIFAR100_Net, CIFAR10_Net, MedNet


def load_data_from_args(args: SimpleArgumentParser) -> Loaders:
    return load_data(args)


def load_data(args):

    if args.dataset == 'mnist':
        dataset_train = Custom_MNIST(few_shot_percent=args.few_shot_train_percent, critic_samples=None,
                                     root='data/', train=True, transform=None, download=True)
        dataset_train_critic = Custom_MNIST(few_shot_percent=args.few_shot_train_percent,
                                            critic_samples=args.n_critic_batches * args.batch_size_critic,
                                            root='data/', train=True, transform=None, download=True)
        dataset_test = Custom_MNIST(few_shot_percent=1.,
                                    critic_samples=None, root='data/', train=False, transform=None, download=True)
        global_vars.N_CLASS = 10
    elif args.dataset == 'decoymnist':
        dataset_train = Custom_DecoyMNIST(few_shot_percent=args.few_shot_train_percent, critic_samples=None,
                                     root='data/', train=True, transform=None, download=True)
        dataset_train_critic = Custom_DecoyMNIST(few_shot_percent=args.few_shot_train_percent,
                                            critic_samples=args.n_critic_batches * args.batch_size_critic,
                                            root='data/', train=True, transform=None, download=True)
        dataset_test = Custom_DecoyMNIST(few_shot_percent=1.,
                                    critic_samples=None, root='data/', train=False, transform=None,
                                    download=True)
        global_vars.N_CLASS = 10
    elif args.dataset == 'decoymnist-unconf':
        dataset_train = Custom_DecoyMNIST_unconf(few_shot_percent=args.few_shot_train_percent,
                                                 critic_samples=args.n_critic_batches * args.batch_size_critic,
                                                 root='data/', mode='train', transform=None, download=True)
        # dataset_train_critic = Custom_DecoyMNIST_unconf(few_shot_percent=1.,
        #                                     critic_samples=args.n_critic_batches * args.batch_size_critic,
        #                                     root='data/', mode='critic', transform=None, download=True)
        dataset_train_critic = Custom_MNIST(few_shot_percent=args.few_shot_train_percent,
                                            critic_samples=args.n_critic_batches * args.batch_size_critic,
                                            root='data/', train=True, transform=None, download=True)
        dataset_test = Custom_DecoyMNIST_unconf(few_shot_percent=1.,
                                                # critic_samples=args.n_critic_batches * args.batch_size_critic,
                                                critic_samples=0,
                                                root='data/', mode='test', transform=None, download=True)
        global_vars.N_CLASS = 10
    elif args.dataset == 'decoymnist-unconf-baseline':
        dataset_train = Custom_DecoyMNIST_unconf(few_shot_percent=args.few_shot_train_percent,
                                                 critic_samples=args.n_critic_batches * args.batch_size_critic,
                                                 root='data/', mode='train', transform=None, download=True)
        dataset_train_critic = Custom_MNIST(few_shot_percent=args.few_shot_train_percent,
                                            critic_samples=args.n_critic_batches * args.batch_size_critic,
                                            root='data/', train=True, transform=None, download=True)
        train_class_weights = dataset_train.class_weights
        dataset_train = torch.utils.data.ConcatDataset([dataset_train, dataset_train_critic])
        dataset_train.class_weights = train_class_weights

        dataset_test = Custom_DecoyMNIST_unconf(few_shot_percent=1.,
                                                critic_samples=args.n_critic_batches * args.batch_size_critic,
                                                root='data/', mode='test', transform=None, download=True)
        global_vars.N_CLASS = 10

    elif args.dataset == 'chestmnist':
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        dataset_train = Custom_ChestMNIST(few_shot_percent=args.few_shot_train_percent, critic_samples=None,
                                              root='data/', split='train', transform=data_transform, download=True)
        dataset_train_critic = Custom_ChestMNIST(few_shot_percent=args.few_shot_train_percent,
                                                     critic_samples=args.n_critic_batches * args.batch_size_critic,
                                                     root='data/', split='train', transform=data_transform,
                                                     download=True)
        dataset_test = Custom_ChestMNIST(few_shot_percent=1., split='test',
                                             critic_samples=None, root='data/', transform=data_transform, download=True)
        global_vars.N_CLASS = 2
    elif args.dataset == 'colormnist':
        data_dir_cmnist = 'data/cmnist/'

        dataset_train = Custom_ColorMNIST(few_shot_percent=args.few_shot_train_percent, critic_samples=None,
                                          root=data_dir_cmnist, split='train')

        dataset_train_critic = Custom_ColorMNIST(few_shot_percent=args.few_shot_train_percent,
                                                     critic_samples=args.n_critic_batches * args.batch_size_critic,
                                                     root=data_dir_cmnist, split='train')
        dataset_test = Custom_ColorMNIST(few_shot_percent=1., critic_samples=None,
                                          root=data_dir_cmnist, split='test')
        global_vars.N_CLASS = 10
    elif args.dataset == 'colormnist-unconf':
        data_dir_cmnist = 'data/cmnist/'

        dataset_train = Custom_ColorMNIST_unconf(critic_samples=args.n_critic_batches * args.batch_size_critic,
                                                 root=data_dir_cmnist, mode='train')

        dataset_train_critic = Custom_ColorMNIST_unconf(critic_samples=args.n_critic_batches * args.batch_size_critic,
                                                 root=data_dir_cmnist, mode='critic')
        dataset_test = Custom_ColorMNIST_unconf(critic_samples=args.n_critic_batches * args.batch_size_critic,
                                                root=data_dir_cmnist, mode='test')
        global_vars.N_CLASS = 10
    elif args.dataset == 'colormnist-unconf-baseline':
        data_dir_cmnist = 'data/cmnist/'

        dataset_train = Custom_ColorMNIST_unconf(critic_samples=args.n_critic_batches * args.batch_size_critic,
                                                 root=data_dir_cmnist, mode='train')
        dataset_train_critic = Custom_ColorMNIST_unconf(critic_samples=args.n_critic_batches * args.batch_size_critic,
                                                 root=data_dir_cmnist, mode='critic')
        train_class_weights = dataset_train.class_weights
        dataset_train = torch.utils.data.ConcatDataset([dataset_train, dataset_train_critic])
        dataset_train.class_weights = train_class_weights

        dataset_test = Custom_ColorMNIST_unconf(critic_samples=args.n_critic_batches * args.batch_size_critic,
                                                root=data_dir_cmnist, mode='test')
        global_vars.N_CLASS = 10

    else:
        exit()

    if args.sep_critic_set:
        n_critic_samples = args.n_critic_batches * args.batch_size_critic
        n_training_samples = len(dataset_train) - n_critic_samples
        assert n_training_samples >= 0
        # split training set into one training set for the classification, and one for the critic
        train_split = [n_training_samples, n_critic_samples]
        dataset_train, dataset_train_critic = random_split(dataset_train, train_split)

    print(f"{args.dataset} data loaded!")
    print(f"Training data loaded {len(dataset_train)} samples")
    if args.sep_critic_set:
        print(f"Training critic data loaded {len(dataset_train_critic)} separate samples")
    else:
        print(f"Training critic data loaded {len(dataset_train_critic)} included samples")
    print(f"Test data loaded {len(dataset_test)} samples")

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, num_workers=0, shuffle=True
    )
    critic_train_loader = torch.utils.data.DataLoader(
        dataset_train_critic, batch_size=args.batch_size_critic, num_workers=0, shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.test_batch_size, num_workers=0, shuffle=False,
    )

    args.n_training_batches = len(train_loader)
    args.n_test_batches = len(test_loader)
    global_vars.LOGGING.n_training_batches = args.n_training_batches
    global_vars.LOGGING.n_test_batches = args.n_test_batches

    viz_dataloader = get_viz_test_data(args, dataset_test, batch_size=args.batch_size)

    loaders = Loaders(train_loader, critic_train_loader, test_loader, viz_dataloader)

    return loaders


def get_viz_test_data(args, dataset_test, batch_size, n_samples=40):
    # visualization_sets = []
    viz_data = []
    viz_labels = []
    try:
        class_ids = np.unique(dataset_test.labels)
    except:
        class_ids = torch.unique(dataset_test.targets)
    for label in range(len(class_ids)):
        try:
            ids = torch.where(dataset_test.targets == label)[0][:n_samples // 10]
        except:
            ids = np.where(dataset_test.labels == label)[0][:n_samples // 10]
        viz_labels.append(torch.ones(n_samples // 10)*label)
        try:
            viz_data.append(dataset_test.data[ids])
        except:
            viz_data.append(dataset_test.imgs[ids])
    try:
        visualization_set = TensorDataset(torch.cat(viz_data, dim=0), torch.cat(viz_labels, dim=0).type(torch.int64))
    except:
        data = torch.tensor(np.concatenate(viz_data, axis=0)).float().div(255)
        data = torch.moveaxis(data, [0, 1, 2, 3], [0, 2, 3, 1])
        labels = torch.tensor(np.concatenate(viz_labels, axis=0)).type(torch.int64)
        visualization_set = TensorDataset(data, labels)

    return DataLoader(visualization_set, batch_size=len(visualization_set))


def get_model_fn(args):
    if args.model == 'SimpleConvNet':
        learner_model_fn = SimpleConvNet
        critic_model_fn = SimpleConvNet
    elif args.model == 'Net1':
        learner_model_fn = Net
        critic_model_fn = Net
    elif args.model == 'MedNet1':
        learner_model_fn = MedNet
        critic_model_fn = MedNet
    elif args.model == 'MLP':
        learner_model_fn = MLP
        critic_model_fn = MLP
    elif args.model == 'Net1-MLP':
        learner_model_fn = Net
        critic_model_fn = MLP
    elif args.model == 'MLP-Net1':
        learner_model_fn = MLP
        critic_model_fn = Net
    return learner_model_fn, critic_model_fn


def colored(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def set_device(args: SimpleArgumentParser):
    if not torch.cuda.is_available() or args.no_cuda:
        print(colored(200, 150, 0, f"No GPU found, falling back to CPU."))
        global_vars.DEVICE = "cpu"
    else:
        global_vars.DEVICE = "cuda"


def set_mean_std(args: SimpleArgumentParser):
    if 'fmnist' in args.dataset or 'derma' in args.dataset or \
            'pneumonia' in args.dataset or 'chest' in args.dataset:
        global_vars.MEAN = 0.5
        global_vars.STD = 0.5
    elif 'mnist' in args.dataset:
        global_vars.MEAN = 0.1307
        global_vars.STD = 0.3081


def set_sharing_strategy():
    # The following prevents there being too many open files at dl1.
    torch.multiprocessing.set_sharing_strategy('file_system')


def write_config_to_log(args: SimpleArgumentParser, log_dir):
    # Write config to log file
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), 'w') as f:
        json_dump: str = json.dumps(args.__dict__, default=lambda o: '<not serializable>')
        f.write(json_dump)
    # TODO: use typed_argparse's save.


def date_time_string():
    date_time = str(datetime.now())[0:-7]
    return date_time.replace(" ", "_")


def config_string(cfg: SimpleArgumentParser) -> str:
    lr_mode = "_sched" if cfg.lr_scheduling else ""

    # just for somewhat nicer formatting:
    run_name = cfg.run_name + "_" if cfg.run_name else ""

    return f'{run_name}' \
           f'{cfg.model}' \
           f'_{cfg.explanation_mode}' \
           f'_seed{cfg.random_seed}' \
           f'_dataset_{cfg.dataset}' \
           f'_{cfg.training_mode}_cr{cfg.n_critic_batches}' \
           f'_lr{cfg.learning_rate}{lr_mode}' \
           f'_bs{cfg.batch_size}_ep{cfg.n_epochs}_p-ep{cfg.n_pretraining_epochs}' \
           f'_gm{cfg.learning_rate_step}' \
           f'_lr-c{cfg.learning_rate_critic}' \
           f'_lambda{cfg.explanation_loss_weight}' \
           f'_lambdaft{cfg.explanation_loss_weight_finetune}' \
           f'_lambdacls{cfg.classification_loss_weight}' \
           f'_fs{cfg.few_shot_train_percent}' \
           f'_sep{cfg.sep_critic_set}' \
           f'_{date_time_string()}'


def get_one_batch_of_images(loader: DataLoader[Any]) -> Tuple[Tensor, Tensor]:
    images, labels = next(iter(loader))
    images, labels = images.to(global_vars.DEVICE), labels.to(global_vars.DEVICE)
    return images, labels


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args(overriding_args: Optional[List]) -> SimpleArgumentParser:
    args = SimpleArgumentParser()
    if overriding_args is not None:
        args.parse_args(overriding_args)
    else:
        args.parse_args()
    print(args.dataset)
    return args


def setup(args: SimpleArgumentParser, eval_mode: bool = False) -> None:
    set_seed(args.random_seed)
    set_sharing_strategy()
    set_device(args)
    set_mean_std(args)

    if args.training_mode == 'only_classification':
        args.sep_critic_set == False
    if args.training_mode == "test" or args.training_mode == "faithfulness" or args.training_mode == "save_expls":
        args.logging_disabled = True

    if args.logging_disabled:
        writer = None
    else:
        log_dir = f"./runs/{config_string(args)}"
        write_config_to_log(args, log_dir)
        writer = SummaryWriter(log_dir)

    global_vars.LOGGING = Logging(writer, args.run_name, args.log_interval, args.log_interval_accuracy,
                                  args.log_interval_critic)


def simple_plot(data):
    fig, axs = plt.subplots(10, 10, figsize=(10,10))
    for i, ax in enumerate(axs.reshape(-1)):
        ax.imshow(data[i], cmap='gray')
        ax.axis('off')
    return fig


def get_initial_expl_imgs(test_batch_to_visualize, learner, ImageHandler):
    # visualize initial explanations
    ImageHandler.add_input_images(test_batch_to_visualize[0])  # needs only the images, not the labels
    ImageHandler.add_gradient_images(test_batch=test_batch_to_visualize, learner=learner,
                                     additional_caption="0: before training")



def get_git_root() -> str:
    current_path = os.path.dirname(os.path.realpath(__file__))
    git_repo = git.Repo(current_path, search_parent_directories=True)
    return git_repo.git.rev_parse("--show-toplevel")


def compute_accuracy(classifier: nn.Module,
                     data: Union[DataLoader, List[List[Tensor]]],
                     n_batches: Optional[int] = None):
    if n_batches is None:
        n_batches = len(data)
    n_correct_samples: int = 0
    n_test_samples_total: int = 0
    classifier.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data):
            if i >= n_batches:  # only test on a set of the test set size, even for training accuracy.
                break

            inputs, labels = inputs.to(global_vars.DEVICE), labels.to(global_vars.DEVICE)

            outputs = classifier(inputs)

            # the class with the highest output is what we choose as prediction
            _, predicted = torch.max(outputs.data, dim=1)
            n_test_samples_total += labels.size()[0]
            n_correct_samples += (predicted == labels).sum().item()
    total_accuracy = n_correct_samples / n_test_samples_total
    classifier.train()
    return total_accuracy


def loader_to_tensors(dataloader: DataLoader) -> Tuple[Tensor, Tensor]:
    all_input_batches = []
    all_label_batches = []
    for input_batch, label_batch in dataloader:
        all_input_batches.append(input_batch)
        all_label_batches.append(label_batch)
    input_tensor = torch.flatten(torch.stack(all_input_batches), start_dim=0, end_dim=1)
    label_tensor = torch.flatten(torch.stack(all_label_batches), start_dim=0, end_dim=1)
    return input_tensor, label_tensor


# following two methods are from https://github.com/ml-research/A-Typology-to-Explore-and-Guide-Explanatory-Interactive-Machine-Learning/blob/19a5dea06f0237593947518e73da911df0a2992b/util.py#L1
def norm_saliencies_fast(A, positive_only=False):
    """
    Normalize tensor to [0,1] across first dimension (for every batch_i) according to formula
    t(i) = (i_t - min_t) /(max_t + 1e-6).
    Add small constant to prevent zero divison.

    Args:
        A: tensor of shape (n, c, h, w).
        positive_only: if True then take only positive values into account and
            zero out negative values.
    """
    shape = A.shape
    A = A.view(A.size(0), -1)
    if positive_only:
        A[A < 0] = 0.
    A -= A.min(1, keepdim=True)[0]
    A /= (A.max(1, keepdim=True)[0] + 1e-6)  # add small constant preventing zero divison
    A = A.view(shape)
    return A


def get_last_conv_layer(model):
    """Return last conv layer of a pytorch model."""
    index = None
    modules = [module for module in model.modules() if not isinstance(module, nn.Sequential)][1:]
    for i, module in enumerate(modules):
        if 'Conv' in str(module):
            index = i
    if index is None:
        raise Exception("Model has no conv layer!")
    return modules[index]


class Custom_MNIST(MNIST):
    # code snippet from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist

    MEAN: float = 0.1307
    STD: float = 0.3081

    def __init__(self, few_shot_percent=1.0, critic_samples=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # calculate how many samples should be trained on
        self.few_shot_percent = few_shot_percent
        n_samples = int(self.few_shot_percent * len(self.targets))

        self.class_weights = torch.unique(self.targets, return_counts=True)[1]/len(self.targets)

        # take only few_shot_percent from full dataset
        self.data = self.data[:n_samples]
        self.targets = self.targets[:n_samples]

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(self.MEAN).div_(self.STD)

        # # Put both data and targets on GPU in advance
        # self.data, self.targets = self.data.to(global_vars.DEVICE), self.targets.to(global_vars.DEVICE)

        if critic_samples:
            print(n_samples)
            print(critic_samples)
            assert n_samples >= critic_samples
            inds = np.random.choice(len(self.targets), critic_samples, replace=False)
            self.data = self.data[inds]
            self.targets = self.targets[inds]


    def un_normalize(self):
        self.data = self.data.mul_(self.STD).add_(self.MEAN)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class Custom_ChestMNIST(ChestMNIST):
    # code snippet from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist

    def __init__(self, few_shot_percent=1.0, critic_samples=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # calculate how many samples should be trained on
        self.few_shot_percent = few_shot_percent
        n_samples = int(self.few_shot_percent * len(self.labels))

        self.imgs = np.expand_dims(self.imgs, axis=-1)

        self.labels = self.process_labels()

        self.class_weights = torch.tensor(np.unique(self.labels, return_counts=True)[1]/len(self.labels))

        # take only few_shot_percent from full dataset
        self.imgs = self.imgs[:n_samples]
        self.labels = self.labels[:n_samples].squeeze()

        if critic_samples:
            assert n_samples >= critic_samples
            inds = np.random.choice(len(self.labels), critic_samples, replace=False)
            self.imgs = self.imgs[inds]
            self.labels = self.labels[inds]

    def process_labels(self):
        tmp = np.ones(len(self.labels))
        tmp[np.where(np.all(self.labels == 0, axis=1))[0]] = 0.
        return tmp

    def __getitem__(self, index):
        '''
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        '''
        img, target = self.imgs[index], self.labels[index].astype(int)
        # img = Image.fromarray(np.uint8(img * 255), 'L')

        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class Custom_DecoyMNIST(MNIST):
    # code snippet from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist

    MEAN: float = 0.1307
    STD: float = 0.3081

    def __init__(self, few_shot_percent=1.0, critic_samples=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        _, X, y, _, _, _, Xt, yt, _, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data/Custom_DecoyMNIST/decoy-mnist.npz')

        if self.train:
            self.data = torch.tensor(X).reshape(X.shape[0], 28, 28)
            self.targets = torch.tensor(y).to(torch.int64)
        else:
            self.data = torch.tensor(Xt).reshape(Xt.shape[0], 28, 28)
            self.targets = torch.tensor(yt).to(torch.int64)

        # calculate how many samples should be trained on
        self.few_shot_percent = few_shot_percent
        n_samples = int(self.few_shot_percent * len(self.targets))

        self.class_weights = torch.unique(self.targets, return_counts=True)[1]/len(self.targets)

        # take only few_shot_percent from full dataset
        self.data = self.data[:n_samples]
        self.targets = self.targets[:n_samples]

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(self.MEAN).div_(self.STD)

        if critic_samples:
            assert n_samples >= critic_samples
            inds = np.random.choice(len(self.targets), critic_samples, replace=False)
            self.data = self.data[inds]
            self.targets = self.targets[inds]


    def un_normalize(self):
        self.data = self.data.mul_(self.STD).add_(self.MEAN)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class Custom_DecoyMNIST_unconf(MNIST):
    # code snippet from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist

    MEAN: float = 0.1307
    STD: float = 0.3081

    def __init__(self, mode='train', few_shot_percent=1.0, critic_samples=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        _, X, y, _, _, _, Xt, yt, _, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data/Custom_DecoyMNIST_unconf/decoy-mnist.npz')

        train_data = torch.tensor(X).reshape(X.shape[0], 28, 28)
        train_targets = torch.tensor(y).to(torch.int64)
        test_data = torch.tensor(Xt).reshape(Xt.shape[0], 28, 28)
        test_targets = torch.tensor(yt).to(torch.int64)

        # extract unconfounded critic samples
        assert len(test_targets) >= critic_samples*2
        inds = np.random.choice(len(test_targets), critic_samples, replace=False)
        critic_data = test_data[inds]
        critic_targets = test_targets[inds]

        test_inds = list(set(np.arange(0, len(test_targets))) - set(inds))
        test_data = test_data[test_inds]
        test_targets = test_targets[test_inds]

        if mode == 'train':
            self.data = train_data
            self.targets = train_targets.to(torch.int64)
            self.train = True
        if mode == 'critic':
            self.data = critic_data
            self.targets = critic_targets.to(torch.int64)
            self.train = True
        elif mode == 'test':
            self.data = test_data
            self.targets = test_targets.to(torch.int64)
            self.train = False

        # calculate how many samples should be trained on
        self.few_shot_percent = few_shot_percent
        n_samples = int(self.few_shot_percent * len(self.targets))

        self.class_weights = torch.unique(train_targets, return_counts=True)[1] / len(self.targets)

        # take only few_shot_percent from full dataset
        self.data = self.data[:n_samples]
        self.targets = self.targets[:n_samples]

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(self.MEAN).div_(self.STD)

    def un_normalize(self):
        self.data = self.data.mul_(self.STD).add_(self.MEAN)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class Custom_ColorMNIST_unconf(torch.utils.data.Dataset):
    MEAN: float = 0.1307
    STD: float = 0.3081

    def __init__(self, critic_samples=100, root='data/cmnist/', mode='train'):
        super().__init__()
        assert mode in ['train', 'test', 'critic']
        self.mode = mode

        self.root = root
        train_data = torch.from_numpy(np.load(self.root + 'train_x.npy')).type('torch.FloatTensor')
        train_targets = torch.from_numpy(np.load(self.root + 'train_y.npy')).type('torch.FloatTensor')
        test_data = torch.from_numpy(np.load(self.root + 'test_x.npy')).type('torch.FloatTensor')
        test_targets = torch.from_numpy(np.load(self.root + 'test_y.npy')).type('torch.FloatTensor')

        # extract unconfounded critic samples
        assert len(test_targets) >= critic_samples*2
        inds = np.random.choice(len(test_targets), critic_samples, replace=False)
        critic_data = test_data[inds]
        critic_targets = test_targets[inds]

        test_inds = list(set(np.arange(0, len(test_targets))) - set(inds))
        test_data = test_data[test_inds]
        test_targets = test_targets[test_inds]

        if mode == 'train':
            self.data = train_data
            self.targets = train_targets.to(torch.int64)
            self.train = True
        if mode == 'critic':
            self.data = critic_data
            self.targets = critic_targets.to(torch.int64)
            self.train = True
        elif mode == 'test':
            self.data = test_data
            self.targets = test_targets.to(torch.int64)
            self.train = False

        # convert rgb to grayscale
        self.data = self.convert_rgb_to_gray()
        self.class_weights = torch.unique(train_targets, return_counts=True)[1] / len(self.targets)


    def convert_rgb_to_gray(self):
        data_gray = torch.empty([self.data.shape[0], 1, self.data.shape[2], self.data.shape[3]])
        for idx in range(self.data.shape[0]):
            data_gray[idx] = torchvision.transforms.functional.rgb_to_grayscale(self.data[idx])
        return data_gray


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        return img, target


    def __len__(self):
        return len(self.targets)


class Custom_ColorMNIST(torch.utils.data.Dataset):
    def __init__(self, few_shot_percent=1.0, critic_samples=None, root='data/cmnist/', split='train'):
        super().__init__()
        assert split in ['train', 'test', 'val']

        self.root = root
        data_x = np.load(self.root + f'{split}_x.npy')
        data_y = np.load(self.root + f'{split}_y.npy')

        self.data = torch.from_numpy(data_x).type('torch.FloatTensor')
        self.targets = torch.from_numpy(data_y).type('torch.LongTensor')

        # convert rgb to grayscale
        self.data = self.convert_rgb_to_gray()

        # calculate how many samples should be trained on
        self.few_shot_percent = few_shot_percent
        n_samples = int(self.few_shot_percent * len(self.targets))

        self.class_weights = torch.unique(self.targets, return_counts=True)[1]/len(self.targets)

        # take only few_shot_percent from full dataset
        self.data = self.data[:n_samples]
        self.targets = self.targets[:n_samples]

        # # Scale data to [0,1]
        # self.data = self.data.unsqueeze(1).float().div(255)

        # # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(self.MEAN).div_(self.STD)

        # # Put both data and targets on GPU in advance
        # self.data, self.targets = self.data.to(global_vars.DEVICE), self.targets.to(global_vars.DEVICE)

        if critic_samples:
            assert n_samples >= critic_samples
            inds = np.random.choice(len(self.targets), critic_samples, replace=False)
            self.data = self.data[inds]
            self.targets = self.targets[inds]


    def convert_rgb_to_gray(self):
        data_gray = torch.empty([self.data.shape[0], 1, self.data.shape[2], self.data.shape[3]])
        for idx in range(self.data.shape[0]):
            data_gray[idx] = torchvision.transforms.functional.rgb_to_grayscale(self.data[idx])
        return data_gray


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        return img, target


    def __len__(self):
        return len(self.targets)