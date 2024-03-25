from typing import Tuple, List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy

import utils as utils
import global_vars
from learner import Learner


def compute_encs(model_expl, model_sim, loaders):
    encs_all = []
    labels_all = []
    # (inputs, labels) = next(iter(loaders.test))
    for n_current_batch, (inputs, labels) in enumerate(loaders.test):
    #
    #     outputs = learner.classifier(inputs)

        expls = model_expl.get_explanation_batch(inputs, labels)

        _ = model_sim.classifier(expls)

        encs_all.append(model_sim.classifier.enc.cpu().detach().numpy())
        labels_all.extend(labels.cpu().detach().tolist())

    encs_all = np.concatenate(encs_all, axis=0)
    labels_all = np.array(labels_all)
    return encs_all, labels_all


def compute_intra_inter(model_expl, model_sim, loaders):
    encs_all, labels_all = compute_encs(model_expl, model_sim, loaders)
    n_class = len(np.unique(labels_all))

    # calculate mean encodings and the distance of each sample of a class to its mean encoding
    encs_mean = []
    max_intra = []
    mean_intra = []
    min_inter = []
    # class_id = 0
    for class_id in range(n_class):
        sample_class_ids = np.where(labels_all == class_id)
        sample_no_class_ids = np.where(labels_all != class_id)
        # compute mean encoding per class
        enc_mean = np.expand_dims(np.mean(encs_all[sample_class_ids], axis=0), axis=0)
        encs_mean.append(enc_mean)
        # compute distance of each sample encoding in class to mean encoding
        class_dist = scipy.spatial.distance.cdist(encs_all[sample_class_ids], enc_mean)
        max_intra.append(np.max(class_dist))
        mean_intra.append(np.mean(class_dist))
        # compute distance of each sample encoding in class to sample encodings of other classes
        min_inter.append(np.min(scipy.spatial.distance.cdist(encs_all[sample_class_ids],
                                                             encs_all[sample_no_class_ids])))

    encs_mean = np.array(encs_mean)

    min_avg_inter_max_intra = []
    mean_avg_inter_max_intra = []
    mean_avg_inter_mean_intra = []
    min_inter_max_intra = []
    for class_id in range(n_class):
        mean_dists = scipy.spatial.distance.cdist(encs_mean[np.where(class_id != np.arange(0, n_class))].squeeze(axis=1),
                                                  encs_mean[class_id])

        min_avg_inter_max_intra.append(max_intra[class_id] / np.min(mean_dists))
        mean_avg_inter_mean_intra.append(mean_intra[class_id] / np.mean(mean_dists))
        mean_avg_inter_max_intra.append(max_intra[class_id] / np.mean(mean_dists))
        min_inter_max_intra.append(max_intra[class_id] / min_inter[class_id])

    return min_avg_inter_max_intra, mean_avg_inter_mean_intra, mean_avg_inter_max_intra, min_inter_max_intra


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

    # base model trained only for classification, used for obtaining encodings
    base_model = Learner(learner_model_fn, critic_model_fn, loaders,
                         optimizer_type=args.optimizer,
                         test_batch_to_visualize=test_batch_to_visualize,
                         model_path='None',
                         explanation_mode=args.explanation_mode)
    base_model.load_state(args.misc_model_pt)

    print('All models loaded')


    (sel_min_avg_inter_max_intra, sel_mean_avg_inter_mean_intra,
     sel_mean_avg_inter_max_intra, sel_min_inter_max_intra) = compute_intra_inter(
        learner, base_model, loaders
    )

    (vanilla_min_avg_inter_max_intra, vanilla_mean_avg_inter_mean_intra,
     vanilla_mean_avg_inter_max_intra, vanilla_min_inter_max_intra) = compute_intra_inter(
        vanilla_model, base_model, loaders
    )

    print("Vanilla: ")
    # print(np.mean(vanilla_min_avg_inter_max_intra))
    print(np.mean(vanilla_mean_avg_inter_mean_intra))
    # print(np.mean(vanilla_mean_avg_inter_max_intra))
    # print(np.mean(vanilla_min_inter_max_intra))

    print("SEL: ")
    # print(np.mean(sel_min_avg_inter_max_intra))
    print(np.mean(sel_mean_avg_inter_mean_intra))
    # print(np.mean(sel_mean_avg_inter_max_intra))
    # print(np.mean(sel_min_inter_max_intra))


if __name__ == '__main__':
    main()
