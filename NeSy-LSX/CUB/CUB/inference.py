"""
Evaluate trained models on the official CUB test set
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from CUB.dataset import load_data
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES
from analysis import AverageMeter, multiclass_metric, accuracy, binary_accuracy

K = [1, 3, 5] #top k class accuracies to compute
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def eval(args, use_encoding=False):
    """
    Run inference using model (and model2 if bottleneck)
    Returns: (for notebook analysis)
    all_class_labels: flattened list of class labels for each image
    topk_class_outputs: array of top k class ids predicted for each image. Shape = size of test set * max(K)
    all_class_outputs: array of all logit outputs for class prediction, shape = N_TEST * N_CLASS
    all_attr_labels: flattened list of labels for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    all_attr_outputs: flatted list of attribute logits (after ReLU/ Sigmoid respectively) predicted for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    all_attr_outputs_sigmoid: flatted list of attribute logits predicted (after Sigmoid) for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    wrong_idx: image ids where the model got the wrong class prediction (to compare with other models)
    """
    if args.model_dir:
        model = torch.load(args.model_dir, map_location=device)
        model.encodings = use_encoding
    else:
        model = None

    if not hasattr(model, 'use_sigmoid'):
        if args.use_sigmoid:
            model.use_sigmoid = True
        else:
            model.use_sigmoid = False
    model.eval()
    model.to(device)

    if args.model_dir2:
        model2 = torch.load(args.model_dir2, map_location=device)
        if not hasattr(model2, 'use_sigmoid'):
            if args.use_sigmoid:
                model2.use_sigmoid = True
            else:
                model2.use_sigmoid = False
        model2.eval()
        model2.to(device)
    else:
        model2 = None

    if args.use_attr and not args.no_img:
        attr_acc_meter = AverageMeter()
    else:
        attr_acc_meter = None

    class_acc_meter = []
    for j in range(len(K)):
        class_acc_meter.append(AverageMeter())

    data_dir = os.path.join(BASE_DIR, args.data_dir, args.eval_data + '.pkl')
    loader = load_data([data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
                       n_class=args.n_imgclasses)
    ata = []
    # all_labels = []
    # for _, batch in enumerate(loader):
    #     inputs, labels = batch
    #     inputs = torch.stack(inputs).t()  # .float()
    #     inputs = torch.flatten(inputs, start_dim=1).float()
    #
    #     all_labels.extend(labels.numpy())
    #     all_data.extend(inputs.numpy())
    #
    # all_labels = np.array(all_labels)
    # all_data = np.array(all_data)
    #
    # import matplotlib.pyplot as plt
    #
    # for class_id in [0, 1, 2, 3]:
    #     # class_id = 1
    #     ids = np.where(all_labels == class_id)[0]
    #     fig = plt.figure()
    #     plt.imshow(all_data[ids])
    #     plt.savefig(f'tmp{c
    # all_dlass_id}.png')

    all_attr_labels, all_attr_outputs, all_attr_outputs_sigmoid = [], [], []
    all_class_labels, all_class_outputs, all_class_logits = [], [], []
    topk_class_outputs = []
    if use_encoding:
        all_encodings = []

    for data_idx, data in enumerate(loader):
        if args.use_attr:
            if args.no_img:  # C -> Y
                inputs, labels = data
                if isinstance(inputs, list):
                    inputs = torch.stack(inputs).t().float()
                inputs = inputs.float()
            else:
                inputs, labels, attr_labels = data
                attr_labels = torch.stack(attr_labels).t()  # N x 112
        else:  # simple finetune
            inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        if use_encoding:
            outputs, encodings = model(inputs)
        else:
            outputs = model(inputs)

        if args.use_attr:
            if args.no_img:  # A -> Y
                class_outputs = outputs
            else:
                if args.bottleneck:
                    attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs]
                    if args.use_sigmoid:
                        attr_outputs = attr_outputs_sigmoid
                    else:
                        attr_outputs = outputs
                    if model2:
                        stage2_inputs = torch.cat(attr_outputs, dim=1)
                        class_outputs = model2(stage2_inputs)
                    else:  # for debugging bottleneck performance without running stage 2
                        class_outputs = torch.zeros([inputs.size(0), N_CLASSES],
                                                    dtype=torch.float64).cuda()  # ignore this
                else:  # cotraining, end2end
                    if args.use_sigmoid:
                        attr_outputs = [torch.nn.Sigmoid()(o) for o in outputs[1:]]
                        attr_outputs_sigmoid = attr_outputs
                    else:
                        attr_outputs = outputs[1:]
                        attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs[1:]]

                    class_outputs = outputs[0]

                for i in range(args.n_attributes):
                    acc = binary_accuracy(attr_outputs_sigmoid[i].squeeze(), attr_labels[:, i])
                    acc = acc.data.cpu().numpy()
                    attr_acc_meter.update(acc, inputs.size(0))

                attr_outputs = torch.cat([o.unsqueeze(1) for o in attr_outputs], dim=1)
                attr_outputs_sigmoid = torch.cat([o for o in attr_outputs_sigmoid], dim=1)
                all_attr_outputs.extend(list(attr_outputs.flatten().data.cpu().numpy()))
                all_attr_outputs_sigmoid.extend(list(attr_outputs_sigmoid.flatten().data.cpu().numpy()))
                all_attr_labels.extend(list(attr_labels.flatten().data.cpu().numpy()))
        else:
            class_outputs = outputs[0]

        _, topk_preds = class_outputs.topk(max(K), 1, True, True)
        _, preds = class_outputs.topk(1, 1, True, True)
        all_class_outputs.extend(list(preds.detach().cpu().numpy().flatten()))
        all_class_labels.extend(list(labels.data.cpu().numpy()))
        all_class_logits.extend(class_outputs.detach().cpu().numpy())
        topk_class_outputs.extend(topk_preds.detach().cpu().numpy())
        if use_encoding:
            all_encodings.extend(list(encodings.data.cpu().numpy()))

        np.set_printoptions(threshold=sys.maxsize)
        class_acc = accuracy(class_outputs, labels, topk=K)  # only class prediction accuracy
        for m in range(len(class_acc_meter)):
            class_acc_meter[m].update(class_acc[m], inputs.size(0))

    all_class_logits = np.vstack(all_class_logits)
    topk_class_outputs = np.vstack(topk_class_outputs)
    # indices in which the correct label is not among the top k predicted labels.
    wrong_idx = np.where(np.sum(topk_class_outputs == np.array(all_class_labels).reshape(-1, 1), axis=1) == 0)[0]

    for j in range(len(K)):
        print('Average top %d class accuracy: %.5f' % (K[j], class_acc_meter[j].avg))

    # if args.use_attr and not args.no_img:  # print some metrics for attribute prediction performance
    #     print('Average attribute accuracy: %.5f' % attr_acc_meter.avg)
    #     all_attr_outputs_int = np.array(all_attr_outputs_sigmoid) >= 0.5
    #
    #     balanced_acc, report = multiclass_metric(all_attr_outputs_int, all_attr_labels)
    #     f1 = f1_score(all_attr_labels, all_attr_outputs_int)
    #     print("Total 1's predicted:", sum(np.array(all_attr_outputs_sigmoid) >= 0.5) / len(all_attr_outputs_sigmoid))
    #     print('Avg attribute balanced acc: %.5f' % (balanced_acc))
    #     print("Avg attribute F1 score: %.5f" % f1)
    #     print(report + '\n')
    if use_encoding:
        return class_acc_meter, attr_acc_meter, all_class_labels, topk_class_outputs, all_class_logits, all_attr_labels, all_attr_outputs, all_attr_outputs_sigmoid, wrong_idx, all_encodings
    else:
        return class_acc_meter, attr_acc_meter, all_class_labels, topk_class_outputs, all_class_logits, all_attr_labels, all_attr_outputs, all_attr_outputs_sigmoid, wrong_idx
        # removal of all_attr_outputs2, potentially removing topk_class_outputs and wrong.idx (so far not used anywhere)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-log_dir', default='.', help='where results are stored')
    parser.add_argument('-model_dir', default=None, help='where the trained models are saved')
    parser.add_argument('-model_dir2', default=None, help='where another trained model are saved (for bottleneck only)')
    parser.add_argument('-eval_data', default='test', help='Type of data (train/ val/ test) to be used')
    parser.add_argument('-use_attr', help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)', action='store_true')
    parser.add_argument('-no_img', help='if included, only use attributes (and not raw imgs) for class prediction', action='store_true')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-data_dir', default='', help='directory to the data used for evaluation')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES, help='whether to apply bottlenecks to only a few attributes')
    parser.add_argument('-n_imgclasses', type=int, default=N_CLASSES,
                        help='whether to apply bottlenecks to only a few attributes')
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    args = parser.parse_args()
    args.batch_size = 16

    print(args)
    y_results, c_results = [], []
    result = eval(args)
    class_acc_meter, attr_acc_meter = result[0], result[1]
    y_results.append(1 - class_acc_meter[0].avg[0].item() / 100.)
    if attr_acc_meter is not None:
        c_results.append(1 - attr_acc_meter.avg.item() / 100.)
    else:
        c_results.append(-1)
    values = (np.mean(y_results), np.mean(c_results))
    output_string = '%.4f %.4f' % values
    print_string = 'Error of y: %.4f, Error of C: %.4f' % values
    print(print_string)

    os.makedirs(args.log_dir, exist_ok=True)
    output = open(os.path.join(args.log_dir, 'results.txt'), 'w')
    output.write(output_string)