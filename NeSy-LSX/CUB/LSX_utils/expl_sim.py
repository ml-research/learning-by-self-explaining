import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy

import LSX_utils.utils as expl_utils


def compute_encs(args, model_expl, model_sim, loader):
    encs_all = []
    labels_all = []
    for i, data in enumerate(loader):

        inputs, labels = data

        if isinstance(inputs, list):
            # inputs = [i.long() for i in inputs]
            inputs = torch.stack(inputs).t() # .float()
        inputs = torch.flatten(inputs, start_dim=1).float().to(args.device)
        labels = labels.to(args.device)

        outputs = model_expl(inputs)
        _, preds = torch.max(outputs.data, dim=1)

        # get explanations of model
        expls = expl_utils.generate_intgrad_captum_table(
            model_expl, inputs, labels, device=args.device
        ).to(torch.float32)

        output = model_sim(expls)

        encs_all.append(output.cpu().detach().numpy())
        labels_all.extend(labels.cpu().detach().tolist())

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

    return np.mean(min_avg_inter_max_intra), np.mean(mean_avg_inter_mean_intra), \
           np.mean(mean_avg_inter_max_intra), np.mean(min_inter_max_intra)


def plot_expls(X, y):
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
    for i, ax in enumerate(axs.flat):
        im = ax.imshow(X[i].reshape((28, 28)), cmap='gray')
        ax.axis('off')
        ax.set_title(y[i])
    cbar = fig.colorbar(im, ax=axs[:, -1], shrink=0.8)
    plt.show()