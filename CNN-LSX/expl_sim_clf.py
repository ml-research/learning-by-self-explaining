from typing import Tuple, List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn import tree

import utils as utils
import global_vars
from learner import Learner


def get_expls(model, loaders, plot=False):
    expls_all = []
    labels_all = []
    # (inputs, labels) = next(iter(loaders.test))
    for n_current_batch, (inputs, labels) in enumerate(loaders.test):
        expls = model.get_explanation_batch(inputs, labels)

        expls_all.append(expls.cpu().detach().numpy())
        labels_all.extend(labels.cpu().detach().tolist())

    expls_all = np.concatenate(expls_all, axis=0)
    labels_all = np.array(labels_all)

    expls_all = expls_all.reshape(expls_all.shape[0], -1)

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
    args = utils.parse_args(overriding_args)
    args.logging_disabled = True

    assert args.model_pt is not None
    assert args.vanilla_model_pt is not None

    utils.setup(args)
    print(global_vars.DEVICE)

    # load model trained via sel
    loaders = utils.load_data_from_args(args)
    test_batch_to_visualize = utils.get_one_batch_of_images(loaders.visualization)

    # initialize learner object
    learner_model_fn, critic_model_fn = utils.get_model_fn(args)
    learner = Learner(learner_model_fn, critic_model_fn, loaders,
                      optimizer_type=args.optimizer,
                      test_batch_to_visualize=test_batch_to_visualize,
                      model_path='None',
                      explanation_mode=args.explanation_mode)
    print("Loading and evaluating specified model ...")
    learner.load_state(args.model_pt)

    # base model trained only for classification
    vanilla_model = Learner(learner_model_fn, critic_model_fn, loaders,
                            optimizer_type=args.optimizer,
                            test_batch_to_visualize=test_batch_to_visualize,
                            model_path='None',
                            explanation_mode=args.explanation_mode)
    print("Loading and evaluating specified model ...")
    vanilla_model.load_state(args.vanilla_model_pt)

    print('All models loaded')

    (X_train, y_train), (X_test, y_test) = get_expls(learner, loaders)
    clf = RidgeClassifier().fit(X_train, y_train)
    print(f"SEL RR: {100*clf.score(X_test, y_test)}")
    # clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    # print(f"SEL LR: {100*clf.score(X_test, y_test)}")
    # clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
    # print(f"SEL DT: {100*clf.score(X_test, y_test)}")

    (X_train, y_train), (X_test, y_test) = get_expls(vanilla_model, loaders)
    clf = RidgeClassifier().fit(X_train, y_train)
    print(f"Vanilla RR: {100*clf.score(X_test, y_test)}")
    # clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    # print(f"Vanilla LR: {100*clf.score(X_test, y_test)}")
    # clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
    # print(f"Vanilla DT: {100*clf.score(X_test, y_test)}")



if __name__ == '__main__':
    main()
