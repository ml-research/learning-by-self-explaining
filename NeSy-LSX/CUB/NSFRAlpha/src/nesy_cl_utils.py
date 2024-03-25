import numpy as np
import random
import io
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
# from skimage import color
from sklearn import metrics
from matplotlib import rc
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from captum.attr import InputXGradient, IntegratedGradients, DeepLift, NoiseTunnel

# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})

axislabel_fontsize = 8
ticklabel_fontsize = 8
titlelabel_fontsize = 8

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


def test_hungarian_matching(attrs=torch.tensor([[[0, 1, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0]],
                                                [[0, 1, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0]]]).type(torch.float),
                            pred_attrs=torch.tensor([[[0.01, 0.1, 0.2, 0.1, 0.2, 0.2, 0.01], [0.1, 0.6, 0.8, 0., 0.4, 0.001, 0.9]],
                                                     [[0.01, 0.1, 0.2, 0.1, 0.2, 0.2, 0.01], [0.1, 0.6, 0.8, 0., 0.4, 0.001, 0.9]]]).type(torch.float)):
    hungarian_matching(attrs, pred_attrs, verbose=1)


def hungarian_matching(attrs, preds_attrs, verbose=0):
    """
    Receives unordered predicted set and orders this to match the nearest GT set.
    :param attrs:
    :param preds_attrs:
    :param verbose:
    :return:
    """
    assert attrs.shape[1] == preds_attrs.shape[1]
    assert attrs.shape == preds_attrs.shape
    from scipy.optimize import linear_sum_assignment
    matched_preds_attrs = preds_attrs.clone()
    idx_map_ids = []
    for sample_id in range(attrs.shape[0]):
        # using euclidean distance
        cost_matrix = torch.cdist(attrs[sample_id], preds_attrs[sample_id]).detach().cpu()

        idx_mapping = linear_sum_assignment(cost_matrix)
        # convert to tuples of [(row_id, col_id)] of the cost matrix
        idx_mapping = [(idx_mapping[0][i], idx_mapping[1][i]) for i in range(len(idx_mapping[0]))]

        idx_map_ids.append([idx_mapping[i][1] for i in range(len(idx_mapping))])

        for i, (row_id, col_id) in enumerate(idx_mapping):
            matched_preds_attrs[sample_id, row_id, :] = preds_attrs[sample_id, col_id, :]
        if verbose:
            print('GT: {}'.format(attrs[sample_id]))
            print('Pred: {}'.format(preds_attrs[sample_id]))
            print('Cost Matrix: {}'.format(cost_matrix))
            print('idx mapping: {}'.format(idx_mapping))
            print('Matched Pred: {}'.format(matched_preds_attrs[sample_id]))
            print('\n')
            # exit()

    idx_map_ids = np.array(idx_map_ids)
    return matched_preds_attrs, idx_map_ids


def create_writer(args):
    writer = SummaryWriter(f"runs/srl/{args.name}_seed{args.seed}", purge_step=0)

    writer.add_scalar('Hyperparameters/learningrate', args.lr, 0)
    writer.add_scalar('Hyperparameters/num_epochs', args.epochs, 0)
    writer.add_scalar('Hyperparameters/batchsize', args.batch_size, 0)

    # store args as txt file
    with open(os.path.join(writer.log_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"\n{arg}: {getattr(args, arg)}")
    return writer


def create_expl_images(img, pred_attrs, table_expl_attrs, img_expl, true_class_name, pred_class_name, xticklabels):
    """
    """
    assert pred_attrs.shape[0:2] == table_expl_attrs.shape[0:2]

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 3))
    ax[0].imshow(img)#(pred_attrs * 255).astype(np.uint8))
    ax[0].axis('off')
    ax[0].set_title("Img")

    ax[1].imshow(pred_attrs, cmap='gray')#(pred_attrs * 255).astype(np.uint8))
    # ax[1].axis('off')
    ax[1].set_ylabel('Slot. ID', fontsize=axislabel_fontsize)
    ax[1].yaxis.set_label_coords(-0.1, 0.5)
    ax[1].set_yticks(np.arange(0, 11))
    ax[1].yaxis.set_tick_params(labelsize=axislabel_fontsize)
    ax[1].set_xlabel('Obj. Attr', fontsize=axislabel_fontsize)
    ax[1].set_xticks(range(len(xticklabels)))
    ax[1].set_xticklabels(xticklabels, rotation=90, fontsize=ticklabel_fontsize)
    ax[1].set_title("Pred Attr")

    ax[2].imshow(img_expl)#(pred_attrs * 255).astype(np.uint8))
    ax[2].axis('off')
    ax[2].set_title("Img Expl")

    im = ax[3].imshow(table_expl_attrs)#, cmap='viridis')
    # cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[3])
    # ax[2].axis('off')
    # ax[2].set_ylabel('Obj. ID', fontsize=axislabel_fontsize)
    ax[3].set_yticks(np.arange(0, 11))
    ax[3].yaxis.set_tick_params(labelsize=axislabel_fontsize)
    ax[3].set_xlabel('Obj. Attr', fontsize=axislabel_fontsize)
    ax[3].set_xticks(range(len(xticklabels)))
    ax[3].set_xticklabels(xticklabels, rotation=90, fontsize=ticklabel_fontsize)
    ax[3].set_title("Table Expl")

    fig.suptitle(f"True Class: {true_class_name}; Pred Class: {pred_class_name}", fontsize=titlelabel_fontsize)
    # plt.subplots_adjust(left=0., right=1., bottom=0., top=0.9, wspace=0., hspace=0.)

    return fig


def performance_matrix(true, pred):
    precision = metrics.precision_score(true, pred, average='macro')
    recall = metrics.recall_score(true, pred, average='macro')
    accuracy = metrics.accuracy_score(true, pred)
    f1_score = metrics.f1_score(true, pred, average='macro')
    # print('Confusion Matrix:\n', metrics.confusion_matrix(true, pred))
    print('Precision: {:.3f} Recall: {:.3f}, Accuracy: {:.3f}: ,f1_score: {:.3f}'.format(precision*100,recall*100,
                                                                                         accuracy*100,f1_score*100))
    return precision, recall, accuracy, f1_score


def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title=None,
                          cmap=plt.cm.Blues, sFigName='confusion_matrix.pdf'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
# Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(sFigName)
    return ax


def resize_tensor(input_tensors, h, w):
    input_tensors = torch.squeeze(input_tensors, 1)

    for i, img in enumerate(input_tensors):
        img_PIL = transforms.ToPILImage()(img)
        img_PIL = transforms.Resize([h, w])(img_PIL)
        img_PIL = transforms.ToTensor()(img_PIL)
        if i == 0:
            final_output = img_PIL
        else:
            final_output = torch.cat((final_output, img_PIL), 0)
    final_output = torch.unsqueeze(final_output, 1)
    return final_output
