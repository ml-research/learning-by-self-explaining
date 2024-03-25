import torch
import random
import numpy as np
import os
import argparse
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


axislabel_fontsize = 8
ticklabel_fontsize = 8
titlelabel_fontsize = 8


def create_writer(args):
    writer = SummaryWriter(f"runs/{args.conf_version}/{args.name}_seed{args.seed}", purge_step=0)

    writer.add_scalar('Hyperparameters/learningrate', args.lr, 0)
    writer.add_scalar('Hyperparameters/num_epochs', args.epochs, 0)
    writer.add_scalar('Hyperparameters/batchsize', args.batch_size, 0)

    return writer


def save_args(args):
    # store args as txt file
    with open(os.path.join(args.log_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"\n{arg}: {getattr(args, arg)}")


def add_feedback_to_datasets(list_dataloaders, class_expl_feedback_masks,
                             class_expl_feedback_masks_inv, device):
    for dataloader in list_dataloaders:
        dataloader.dataset.class_expl_feedback_masks = class_expl_feedback_masks.to(device)
        dataloader.dataset.class_expl_feedback_masks_inv = class_expl_feedback_masks_inv.to(device)


def write_clauses(clauses, lsx_iter, log_dir):
    with open(os.path.join(log_dir, "clauses.csv"), "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(clauses)


def read_clauses(log_dir):
    with open(os.path.join(log_dir, "clauses.csv"), newline='') as f:
        reader = csv.reader(f)
        clauses = list(reader)
    return clauses


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


def norm_saliencies(saliencies):
    saliencies_norm = saliencies.clone()

    for i in range(saliencies.shape[0]):
        if len(torch.nonzero(saliencies[i], as_tuple=False)) == 0:
            saliencies_norm[i] = saliencies[i]
        else:
            saliencies_norm[i] = (saliencies[i] - torch.min(saliencies[i])) / \
                                 (torch.max(saliencies[i]) - torch.min(saliencies[i]))

    return saliencies_norm

def generate_intgrad_captum_table(net, x, labels, device="cuda"):
    if x.requires_grad == False:
        x.requires_grad = True

    labels = labels.to(device)
    explainer = IntegratedGradients(net)
    saliencies = explainer.attribute(x, target=labels)
    # remove negative attributions
    saliencies[saliencies < 0] = 0.
    # if reasoner:
        # # normalise the explations and concatenate the object presence prediction back to explanation,
        # # as nsfr requires this information
        # saliencies_norm = torch.cat(
        #     (
        #     torch.ones((len(labels), 1), device=device),
        #     norm_saliencies(saliencies)
        #     ),
        #     dim=1
        # )
    # else:
    saliencies_norm = norm_saliencies(saliencies)
    return saliencies_norm


def create_expl_images(img, pred_attrs, model_symb_expl, img_expl, symb_feedback, true_class_name, pred_class_name,
                       xticklabels):
    """
    """
    assert pred_attrs.shape[0:2] == model_symb_expl.shape[0:2]

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 3))

    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title("Img")

    ax[1].imshow(pred_attrs, cmap='gray')
    ax[1].set_ylabel('Slot. ID', fontsize=axislabel_fontsize)
    ax[1].yaxis.set_label_coords(-0.1, 0.5)
    ax[1].set_yticks(np.arange(0, 11))
    ax[1].yaxis.set_tick_params(labelsize=axislabel_fontsize)
    ax[1].set_xlabel('Obj. Attr', fontsize=axislabel_fontsize)
    ax[1].set_xticks(range(len(xticklabels)))
    ax[1].set_xticklabels(xticklabels, rotation=90, fontsize=ticklabel_fontsize)
    ax[1].set_title("Pred Attr")

    # ax[2].imshow(img_expl)
    # ax[2].axis('off')
    # ax[2].set_title("Img Expl")
    if symb_feedback.shape == model_symb_expl.shape:
        im = ax[2].imshow(symb_feedback)
        ax[2].set_yticks(np.arange(0, 11))
        ax[2].yaxis.set_tick_params(labelsize=axislabel_fontsize)
        ax[2].set_xlabel('Obj. Attr', fontsize=axislabel_fontsize)
        ax[2].set_xticks(range(len(xticklabels)))
        ax[2].set_xticklabels(xticklabels, rotation=90, fontsize=ticklabel_fontsize)
        ax[2].set_title("Model Symb Feedback")

    im = ax[3].imshow(model_symb_expl)
    ax[3].set_yticks(np.arange(0, 11))
    ax[3].yaxis.set_tick_params(labelsize=axislabel_fontsize)
    ax[3].set_xlabel('Obj. Attr', fontsize=axislabel_fontsize)
    ax[3].set_xticks(range(len(xticklabels)))
    ax[3].set_xticklabels(xticklabels, rotation=90, fontsize=ticklabel_fontsize)
    ax[3].set_title("Table Expl")

    fig.suptitle(f"True Class: {true_class_name}; Pred Class: {pred_class_name}", fontsize=titlelabel_fontsize)

    return fig


# def write_expls(net, data_loader, tagname, epoch, writer, args):
#     """
#     Writes NeSy Concpet Learner explanations to tensorboard writer.
#     """
#
#     attr_labels = [
#         'x', 'y', 'z',
#         'Sphere', 'Cube', 'Cylinder',
#         'Large', 'Small',
#         'Rubber', 'Metal',
#         'Cyan', 'Blue', 'Yellow', 'Purple', 'Red', 'Green', 'Gray', 'Brown'
#     ]
#
#     net.eval()
#
#     for i, sample in enumerate(data_loader):
#         # input is either a set or an image
#         imgs, gt_attrs, gt_classes, img_ids, _, _, prop_expls, prop_expls_inv = map(lambda x: x.to(args.device), sample)
#         gt_classes = gt_classes.long()
#
#         # forward evaluation through the network
#         output_cls, output_attrs, obj_pres = net.forward(imgs)
#         _, preds = torch.max(output_cls, 1)
#
#         # get explanations of set classifier
#         model_symb_expls = generate_intgrad_captum_table(net.set_cls, output_attrs, obj_pres, preds, reasoner=False,
#                                                          device=args.device)
#
#         # convert sorting gt target set and gt table explanations to match the order of the predicted table
#         gt_attrs, match_ids = hungarian_matching(output_attrs.to(args.device), gt_attrs)
#         # table_expls = table_expls[:, match_ids][range(table_expls.shape[0]), range(table_expls.shape[0])]
#
#         # get the ids of the two objects that receive the maximal importance, i.e. most important for the classification
#         max_expl_obj_ids = model_symb_expls.max(dim=2)[0].topk(2)[1]
#
#         # get attention masks
#         attns = net.img2state_net.slot_attention.attn
#         # reshape attention masks to 2D
#         attns = attns.reshape((attns.shape[0], attns.shape[1], int(np.sqrt(attns.shape[2])),
#                                int(np.sqrt(attns.shape[2]))))
#
#         # concatenate the visual explanation of the top two objects that are most important for the classification
#         img_saliencies = torch.zeros(attns.shape[0], attns.shape[2], attns.shape[3])
#         for obj_id in range(max_expl_obj_ids.shape[1]):
#             img_saliencies += attns[range(attns.shape[0]), obj_id, :, :].detach().cpu()
#
#         # upscale img_saliencies to orig img shape
#         img_saliencies = resize_tensor(img_saliencies.cpu(), imgs.shape[2], imgs.shape[2]).squeeze(dim=1).cpu()
#
#         for img_id, (img, gt_attr, output_attr, model_symb_expl, img_expl,
#                      true_label, pred_label, imgid, prop_expl, prop_expl_inv) in enumerate(zip(
#                 imgs, gt_attrs, output_attrs, model_symb_expls,
#                 img_saliencies, gt_classes, preds,
#                 img_ids, prop_expls, prop_expls_inv
#         )):
#             # unnormalize images
#             img = img / 2. + 0.5  # Rescale to [0, 1].
#
#             fig = create_expl_images(np.array(transforms.ToPILImage()(img.cpu()).convert("RGB")),
#                                            output_attr.detach().cpu().numpy(),
#                                            model_symb_expl.detach().cpu().numpy(),
#                                            img_expl.detach().cpu().numpy(),
#                                      prop_expl_inv.detach().cpu().numpy(),
#                                            true_label, pred_label, attr_labels)
#             writer.add_figure(f"{tagname}_{img_id}", fig, epoch)
#             if img_id > 10:
#                 break
#
#         break
