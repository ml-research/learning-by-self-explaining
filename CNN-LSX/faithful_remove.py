from typing import Tuple, List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

import utils as utils
import global_vars
from learner import Learner


def run_compute_faithfulness_acc(learner, vanilla_model, loaders, method, mode, plot=False):

    learner_accs, _ = comp_faithfulness_acc(learner, loaders, method=method, mode=mode)
    learner_accs_random, _ = comp_faithfulness_acc(learner, loaders, method=method,
                                               random=True, background=False, mode=mode)
    vanilla_accs, _ = comp_faithfulness_acc(vanilla_model, loaders, method=method, mode=mode)
    vanilla_accs_random, k_range = comp_faithfulness_acc(vanilla_model, loaders, method=method,
                                                     random=True, background=False, mode=mode)

    LSX_base = learner_accs[0]
    vanilla_base = vanilla_accs[0]
    learner_accs = learner_accs[1:]
    vanilla_accs = vanilla_accs[1:]
    learner_accs_random = learner_accs_random[1:]
    vanilla_accs_random = vanilla_accs_random[1:]

    print(f"{mode}:")
    print(f"LSX: {100 * np.mean(np.array(learner_accs_random) - np.array(learner_accs))}")
    print(f"vanilla: {100 * np.mean(np.array(vanilla_accs_random) - np.array(vanilla_accs))}")


def run_compute_faithfulness_prob(learner, vanilla_model, loaders, method, mode, plot=False):

    learner_faithful_score, learner_probs, _ = comp_faithfulness_prob(learner, loaders, method=method, mode=mode)
    learner_faithful_score_random, learner_probs_random, _ = comp_faithfulness_prob(learner, loaders, method=method,
                                               random=True, background=False, mode=mode)
    vanilla_faithful_score, vanilla_probs, _ = comp_faithfulness_prob(vanilla_model, loaders, method=method, mode=mode)
    vanilla_faithful_score_random, vanilla_probs_random, k_range = comp_faithfulness_prob(vanilla_model, loaders, method=method,
                                                     random=True, background=False, mode=mode)

    learner_faithful_score = learner_faithful_score[1:]
    vanilla_faithful_score = vanilla_faithful_score[1:]
    learner_faithful_score_random = learner_faithful_score_random[1:]
    vanilla_faithful_score_random = vanilla_faithful_score_random[1:]

    print(f"{mode}:")
    print(f"LSX: {100 * np.mean(learner_faithful_score_random - learner_faithful_score)}")
    print(f"vanilla: {100 * np.mean(vanilla_faithful_score_random - vanilla_faithful_score)}")


def comp_faithfulness_prob(model, loaders, method='comprehensiveness', random=False, background=False, mode='median'):
    inputs, expls, probs, preds, _, input_mean, input_median, input_min = get_expls(model, loaders, random, background)

    model.classifier.eval()

    B = expls.shape[0]
    W = expls.shape[2]
    H = expls.shape[3]

    expls_flat = expls.view(B, W * H)

    expls_norm = utils.norm_saliencies_fast(expls_flat)

    k_range = 1 - 1/100*np.array([0., 1., 5., 10., 20., 50.])

    faithful_score = []
    probs_ = []
    for k in k_range:
        print(k)
        n_correct_samples: int = 0

        # set the values to 0
        if mode == 'mean':
            value = input_mean
        elif mode == 'median':
            value = input_median
        elif mode == 'min':
            value = input_min
        elif mode == 'zero':
            value = 0.

        tmp_in = torch.clone(inputs).view(B, W * H)
        # get the indices of the top k percent important features and replace them with the specified value
        # _, topk_inds = torch.topk(expls_flat, k=k, dim=1)
        if method == 'comprehensiveness':
            inds = torch.nonzero(expls_norm >= k)
            tmp_in[inds[:, 0], inds[:, 1]] = value
        elif method == 'sufficiency':
            inds = torch.nonzero(expls_norm <= k)
            tmp_in[inds[:, 0], inds[:, 1]] = value

        # make a new prediction with the modified input
        outputs = model.classifier.forward((tmp_in.view(B, 1, W, H)))
        # make a new prediction with the modified input and get probabilities
        probs_k = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        # get the probability of the originally predicted class
        probs_.append(probs_k[torch.arange(B), preds])
        # compute the difference for the originally predicted class probability
        faithful_score.append(np.mean((probs.detach().cpu().numpy() - probs_k[torch.arange(B), preds])))

    return np.array(faithful_score), np.array(probs_), 1. - k_range


def comp_faithfulness_acc(model, loaders, method='comprehensiveness', random=False, background=False, mode='median'):
    inputs, expls, _, preds, labels, input_mean, input_median, input_min = get_expls(model, loaders, random, background)

    model.classifier.eval()

    B = expls.shape[0]
    W = expls.shape[2]
    H = expls.shape[3]

    expls_flat = expls.view(B, W * H)

    expls_norm = utils.norm_saliencies_fast(expls_flat)

    k_range = 1 - 1/100*np.array([0., 1., 5., 10., 20., 50.])

    acc = []
    probs_ = []
    for k in k_range:
        print(k)
        n_correct_samples: int = 0

        # set the values to 0
        if mode == 'mean':
            value = input_mean
        elif mode == 'median':
            value = input_median
        elif mode == 'min':
            value = input_min
        elif mode == 'zero':
            value = 0.

        tmp_in = torch.clone(inputs).view(B, W * H)
        # get the indices of the top k percent important features and replace them with the specified value
        # _, topk_inds = torch.topk(expls_flat, k=k, dim=1)
        if method == 'comprehensiveness':
            inds = torch.nonzero(expls_norm >= k)
            tmp_in[inds[:, 0], inds[:, 1]] = value
        elif method == 'sufficiency':
            inds = torch.nonzero(expls_norm <= k)
            tmp_in[inds[:, 0], inds[:, 1]] = value

        # make a new prediction with the modified input
        outputs = model.classifier.forward((tmp_in.view(B, 1, W, H)))

        _, preds = torch.max(outputs.data, dim=1)
        n_correct_samples += (preds == labels).sum().item()

        total_accuracy = n_correct_samples / len(labels)
        acc.append(total_accuracy)

    return np.array(acc), 1. - k_range


def get_expls(model, loaders, random, background=False):
    expls_all = []
    labels_all = []
    probs_all = []
    preds_all = []
    inputs_all = []
    # (inputs, labels) = next(iter(loaders.test))
    for n_current_batch, (inputs, labels) in enumerate(loaders.test):

        outputs = model.classifier(inputs)
        # the class with the highest output is what we choose as prediction
        probs = torch.softmax(outputs.data, dim=1)
        _, preds = torch.max(outputs.data, dim=1)

        if random:
            expls = torch.rand(inputs.shape)
            if background:
                expls[torch.where(inputs > 0.)] = -1.
        else:
            expls = model.get_explanation_batch(inputs, labels)

        expls_all.append(expls)
        labels_all.extend(labels)
        probs_all.append(probs[torch.arange(len(labels)), preds])
        preds_all.append(preds)
        inputs_all.append(inputs)

    inputs_all = torch.cat(inputs_all, dim=0)
    expls_all = torch.cat(expls_all, dim=0)
    probs_all = torch.cat(probs_all, dim=0)
    preds_all = torch.cat(preds_all, dim=0)
    labels_all = torch.tensor(labels_all)

    input_median = torch.median(inputs_all)
    input_mean = torch.mean(inputs_all)
    input_min = torch.min(inputs_all)

    #     plot_expls(expls_all.detach().cpu().numpy(), labels_all.detach().cpu().numpy())

    return inputs_all, expls_all, probs_all, preds_all, labels_all, input_mean.item(), input_median.item(), \
           input_min.item()


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

    # compute and present the comprehensiveness results
    # run_compute_faithfulness_prob(learner, vanilla_model, loaders, method='comprehensiveness', mode='median')
    run_compute_faithfulness_acc(learner, vanilla_model, loaders, method='comprehensiveness', mode='median')

    # compute and present the sufficiency results
    # run_compute_faithfulness_prob(learner, vanilla_model, loaders, method='sufficiency', mode='median')
    run_compute_faithfulness_acc(learner, vanilla_model, loaders, method='sufficiency', mode='median')


if __name__ == '__main__':
    main()
