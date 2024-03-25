from typing import Tuple, List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy

import NeSyConceptLearner.src.model_for_nsfr as nesycl_model
import NeSyConceptLearner.src.utils as nesycl_utils
import utils as utils
import utils_unconfound as utils_unconfound


def compute_encs(args, model_expl, model_sim, loader):
    encs_all = []
    labels_all = []
    for i, sample in enumerate(loader):

        # input is either a set or an image
        imgs, _, gt_classes, _, _, _, _, _ = map(lambda x: x.to(args.device), sample)
        gt_classes = gt_classes.long()

        # forward evaluation through the network
        output_cls, output_attr, obj_pres = model_expl.forward(imgs)
        _, preds = torch.max(output_cls, 1)

        expls = utils.generate_intgrad_captum_table(model_expl.set_cls, output_attr, obj_pres,
                                                    gt_classes, device=args.device,
                                                    reasoner=False)

        _ = model_sim.set_cls(expls)

        encs_all.append(model_sim.set_cls.encoding.view(expls.shape[0], -1).cpu().detach().numpy())
        labels_all.extend(gt_classes.cpu().detach().tolist())

    encs_all = np.concatenate(encs_all, axis=0)
    labels_all = np.array(labels_all)
    return encs_all, labels_all


def compute_intra_inter(args, model_expl, model_sim, loader):
    encs_all, labels_all = compute_encs(args, model_expl, model_sim, loader)
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

    print(f"Loading misc model used for encoding (see paper for details)")
    fp_ckpt_base = 'runs/.../model.pth'
    base_model, _, _ = utils.load_nn_model(args, fp_ckpt=fp_ckpt_base)

    print('All models loaded')

    (_, lsx_mean_avg_inter_mean_intra, _, _) = compute_intra_inter(
        args, learner, base_model, loaders[2]
    )

    (_, vanilla_mean_avg_inter_mean_intra, _, _) = compute_intra_inter(
        args, vanilla_model, base_model, loaders[2]
    )

    print("Vanilla: ")
    print(np.mean(vanilla_mean_avg_inter_mean_intra))

    print("SEL: ")
    print(np.mean(lsx_mean_avg_inter_mean_intra))


if __name__ == '__main__':
    main()
