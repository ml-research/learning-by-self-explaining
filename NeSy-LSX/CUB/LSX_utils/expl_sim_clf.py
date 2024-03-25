import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier

import LSX_utils.utils as expl_utils

SEED=0
random.seed(SEED)
np.random.seed(SEED)


def get_expls(args, model, loader, plot=False):
    expls_all = []
    labels_all = []

    for i, data in enumerate(loader):

        inputs, labels = data

        if isinstance(inputs, list):
            # inputs = [i.long() for i in inputs]
            inputs = torch.stack(inputs).t() # .float()
        inputs = torch.flatten(inputs, start_dim=1).float().to(args.device)
        labels = labels.to(args.device)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, dim=1)

        # get explanations of model
        expls = expl_utils.generate_intgrad_captum_table(
            model, inputs, labels, device=args.device
        ).unsqueeze(dim=1)

        expls_all.append(expls.cpu().detach().numpy())
        labels_all.extend(labels.cpu().detach().tolist())

    expls_all = np.concatenate(expls_all, axis=0)
    labels_all = np.array(labels_all)

    expls_all = expls_all.reshape(expls_all.shape[0], -1)

    # number of training samples for RR
    n_train_samples = int(0.8 * len(labels_all))

    perm = np.random.permutation(len(labels_all))
    expls_all = expls_all[perm]
    labels_all = labels_all[perm]

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


def comp_sim_clf(model, loader, args):
    (X_train, y_train), (X_test, y_test) = get_expls(args, model, loader)
    clf = RidgeClassifier().fit(X_train, y_train)
    return 100*clf.score(X_test, y_test)