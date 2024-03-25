#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from CUB.inference import *
from CUB.utils import get_attribute_groupings

import rtpt

rtpt = rtpt.RTPT(name_initials="DS", experiment_name="concept bottleneck models", max_iterations=1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def simulate_group_intervention(mode, model2, attr_group_dict,
                                b_attr_binary_outputs, b_class_labels,
                                b_attr_outputs, b_attr_labels, uncertainty_attr_labels, use_not_visible,
                                n_replace):

    assert len(uncertainty_attr_labels) == len(b_attr_labels), \
        'len(uncertainty_attr_labels): %d, len(b_attr_labels): %d' % (len(uncertainty_attr_labels), len(b_attr_labels))

    all_class_acc = []
    # conversion to numpy arrays
    b_attr_labels = np.array(b_attr_labels)
    b_attr_outputs = np.array(b_attr_outputs)
    b_attr_binary_outputs = np.array(b_attr_binary_outputs)
    b_class_labels = np.array(b_class_labels)
    uncertainty_attr_labels = np.array(uncertainty_attr_labels)

    if mode == 'random':
        replace_fn = lambda: replace_random()

        def replace_random():
            replace_idx = []
            group_replace_idx = list(random.sample(list(range(args.n_groups)), n_replace))
            for i in group_replace_idx:
                replace_idx.extend(attr_group_dict[i])
            return np.array(replace_idx)
    else:
        replace_fn = lambda attr_true, attr_pred: replace_diff(attr_true, attr_pred)

        def replace_diff(attr_true, attr_pred):
            """
            Select the n_replace groups which have the largest difference to the true attributions.
            Should they be weighted by the group size?
            """
            if n_replace == 0:
                return np.array([])
            group_errors = np.zeros(args.n_groups)
            for i in range(args.n_groups):
                group_errors[i] = np.sum(np.absolute(attr_pred[attr_group_dict[i]] - attr_true[attr_group_dict[i]]))

            # sort the group indices by group errors
            group_replace_idx = np.argpartition(group_errors, -n_replace)[-n_replace:]
            replace_idx = []
            for i in group_replace_idx:
                replace_idx.extend(attr_group_dict[i])
            return np.array(replace_idx)

    n_trials = 5 if mode == 'random' else 1
    for _ in range(n_trials):
        b_attr_new = np.copy(b_attr_outputs)[:]

        attr_replace_idx = []

        # get a list of all attributes to replace
        for img_id in range(len(b_class_labels)):
            if mode == 'random':
                replace_idx = replace_fn()
            else:   # based on error
                attr_pred = b_attr_outputs[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
                attr_true = b_attr_labels[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
                replace_idx = replace_fn(attr_true, attr_pred)

            attr_replace_idx.extend(replace_idx + img_id * args.n_attributes)

        pred_vals = b_attr_binary_outputs[attr_replace_idx]
        true_vals = b_attr_labels[attr_replace_idx]
        print("acc among the replaced values:", (pred_vals == true_vals).mean())

        # change b_attr_new to the new interventioned values, while respecting invisibility
        b_attr_new[attr_replace_idx] = b_attr_labels[attr_replace_idx]

        if use_not_visible:
            not_visible_idx = np.where(uncertainty_attr_labels == 1)[0]
            for idx in attr_replace_idx:
                if idx in not_visible_idx:
                    b_attr_new[idx] = b_attr_labels[idx]

        # stage 2
        model2.eval()
        model2.to(device)

        b_attr_new = b_attr_new.reshape(-1, args.n_attributes)
        stage2_inputs = torch.from_numpy(b_attr_new).to(device)

        class_outputs = model2(stage2_inputs)

        _, preds = class_outputs.topk(1, 1, True, True)
        b_class_outputs_new = preds.data.cpu().numpy().squeeze()
        class_acc = np.mean(b_class_outputs_new == b_class_labels)
        all_class_acc.append(class_acc * 100)
    return max(all_class_acc)


def run(args):
    # load the test data
    uncertainty_attr_labels = []
    test_data = pickle.load(open(os.path.join(args.data_dir, 'test.pkl'), 'rb'))
    mask = pickle.load(open(os.path.join(args.data_dir, 'mask.pkl'), 'rb'))
    for d in test_data:
        attr_certainty = np.array(d['attribute_certainty'])
        uncertainty_attr_labels.extend(list(attr_certainty[mask]))

    # group the attributes based on their connections - e.g. group beak-color::black, beak-color::brown, ...
    attr_group_dict = get_attribute_groupings()

    # apply the mask to the mapping: only keep the attributes which are not filtered out
    for group_id, attr_ids in attr_group_dict.items():
        new_attr_ids = []
        for attr_id in attr_ids:
            if attr_id in mask:
                new_attr_ids.append(attr_id)
        attr_group_dict[group_id] = new_attr_ids

    # update the enumeration of the attributes in the group mapping to match the filtered data
    total_so_far = 0
    for group_id, attr_ids in attr_group_dict.items():
        class_attr_ids = list(range(total_so_far, total_so_far + len(attr_ids)))
        total_so_far += len(attr_ids)
        attr_group_dict[group_id] = class_attr_ids

    # stage 1
    _, _, b_class_labels, b_topk_class_outputs, b_class_logits, b_attr_labels, b_attr_outputs, b_attr_outputs_sigmoid, \
        b_wrong_idx, encodings = eval(args, use_encoding=True)

    b_attr_binary_outputs = np.rint(b_attr_outputs_sigmoid).astype(int)

    assert args.mode in ['error', 'random']

    # stage 2
    model = torch.load(args.model_dir)
    if args.model_dir2:
        model2 = torch.load(args.model_dir2)
    else:  # end2end, split model into 2
        all_mods = list(model.modules())
        model2 = all_mods[-1]  # last fully connected layer

    model2.to(device)
    results = []
    for n_replace in list(range(args.n_groups + 1)):
        acc = simulate_group_intervention(args.mode,
                                          model2,
                                          attr_group_dict,
                                          b_attr_binary_outputs,
                                          b_class_labels,
                                          b_attr_outputs,
                                          b_attr_labels,
                                          uncertainty_attr_labels,
                                          args.use_invisible,
                                          n_replace)
        print(n_replace, acc)
        results.append([n_replace, acc])
    return results


def parse_arguments(parser=None):
    if parser is None: parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-log_dir', default='.', help='where results are stored')
    parser.add_argument('-model_dir', help='where the trained model is saved')
    parser.add_argument('-model_dir2', default=None, help='where another trained model is saved (for bottleneck only)')
    parser.add_argument('-eval_data', default='test', help='Type of data (val/ test) to be used')
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-use_attr', help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)', action='store_true')
    parser.add_argument('-no_img', help='if included, only use attributes (and not raw imgs) for class prediction', action='store_true')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
    parser.add_argument('-data_dir', default='', help='directory to the data used for evaluation')
    parser.add_argument('-n_attributes', type=int, default=112, help='whether to apply bottlenecks to only a few attributes')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-attribute_group', default=None, help='file listing the (trained) model directory for each attribute group')
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-use_invisible', help='Whether to include attribute visibility information', action='store_true')
    parser.add_argument('-mode', help='Which mode to use for correction. Choose from error, random', default='random')
    parser.add_argument('-n_groups', help='Number of groups', type=int, default=28)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = parse_arguments()
    all_values = []
    # if there are multiple models given, perform the tti on each model.
    values = run(args)
    all_values.append(values)

    output_string = ''
    no_intervention_groups = np.array(all_values[0])[:, 0]
    values = sum([np.array(values)[:, 1] / len(all_values) for values in all_values])
    for no_intervention_group, value in zip(no_intervention_groups, values):
        output_string += '%.1f %.4f\n' % (no_intervention_group, value)
    print(output_string)
    os.makedirs(args.log_dir, exist_ok=True)
    output = open(os.path.join(args.log_dir, 'results.txt'), 'w')
    output.write(output_string)

