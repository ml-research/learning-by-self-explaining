from typing import Tuple, List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn import tree

import NeSyConceptLearner.src.model_for_nsfr as nesycl_model
import NeSyConceptLearner.src.utils as nesycl_utils
import utils as utils

def get_expls(args, model, loader, plot=False):
    expls_all = []
    labels_all = []

    for i, sample in enumerate(loader):

        # input is either a set or an image
        imgs, _, gt_classes, _, _, _, _, _ = map(lambda x: x.to(args.device), sample)
        gt_classes = gt_classes.long()

        # forward evaluation through the network
        output_cls, output_attr, obj_pres = model.forward(imgs)
        _, preds = torch.max(output_cls, 1)

        expls = utils.generate_intgrad_captum_table(model.set_cls, output_attr, obj_pres,
                                            gt_classes, device=args.device,
                                            reasoner=False)

        expls_all.append(expls.cpu().detach().numpy())
        labels_all.extend(gt_classes.cpu().detach().tolist())

    expls_all = np.concatenate(expls_all, axis=0)
    labels_all = np.array(labels_all)

    expls_all = expls_all.reshape(expls_all.shape[0], -1)

    # number of training samples for RR
    n_train_samples = int(0.8 * len(labels_all))

    train_expls_all = expls_all[:n_train_samples]
    test_expls_all = expls_all[n_train_samples:]
    train_labels_all = labels_all[:n_train_samples]
    test_labels_all = labels_all[n_train_samples:]

    if plot:
        plot_expls(train_expls_all, train_labels_all)

    return (train_expls_all, train_labels_all), (test_expls_all, test_labels_all)


def plot_expls(X, y):
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
    for i, ax in enumerate(axs.flat):
        im = ax.imshow(X[i].reshape((28, 28)), cmap='gray')
        ax.axis('off')
        ax.set_title(y[i])
    cbar = fig.colorbar(im, ax=axs[:, -1], shrink=0.8)
    plt.show()


def main(overriding_args: Optional[List] = None):
    print("Setting up experiments...")
    args = utils.get_args()

    assert args.fp_ckpt is not None
    assert args.fp_ckpt_vanilla is not None

    # get data loaders
    print("\nData loading ...")
    loaders = utils.get_dataloaders(args, mode='mix')
    args.m = args.n_imgclasses
    print("Data loaded ...")

    if args.dataset == 'unconfound':
        print("Preparing datasets for unconfounding task ...")
        loaders = utils_unconfound.update_val_and_test_dataset(
            loaders, n_samples_per_class=args.unconfound_n_samples_per_class, args=args
        )

    # initialize learner object
    print(f"Loading {args.fp_ckpt}")
    learner, _, _ = utils.load_nn_model(args, fp_ckpt=args.fp_ckpt)

    # base model trained only for classification
    print(f"Loading {args.fp_ckpt_vanilla}")
    vanilla_model, _, _ = utils.load_nn_model(args, fp_ckpt=args.fp_ckpt_vanilla)

    print('All models loaded')

    (X_train, y_train), (X_test, y_test) = get_expls(args, learner, loaders[2])
    clf = RidgeClassifier().fit(X_train, y_train)
    print(f"SEL RR: {100*clf.score(X_test, y_test)}")

    (X_train, y_train), (X_test, y_test) = get_expls(args, vanilla_model, loaders[2])
    clf = RidgeClassifier().fit(X_train, y_train)
    print(f"Vanilla RR: {100*clf.score(X_test, y_test)}")


if __name__ == '__main__':
    main()
