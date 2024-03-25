import torch
import random
import numpy as np
import os
import argparse
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import data_clevr_hans as data
import data_clevr_hans_bin as data_bin
import NeSyConceptLearner.src.model_for_nsfr as nesycl_model
from captum.attr import IntegratedGradients, InputXGradient, Saliency
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


axislabel_fontsize = 8
ticklabel_fontsize = 8
titlelabel_fontsize = 8


def get_args():
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help="Name to store the log file as",
    )
    parser.add_argument(
        "--mode", type=str, default="train", help="train or test"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="What dataset experiment are you runnning? E.g. unconfound?"
    )
    parser.add_argument("--resume", help="Path to log file to resume from")
    parser.add_argument(
        "--num", type=int, default=10, help="Just number of run"
    )
    parser.add_argument(
        "--seed", type=int, default=10, help="Random generator seed for all frameworks"
    )
    parser.add_argument(
        "--lsx-iters", type=int, default=10, help="Number of LSX iterations"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--expl-epochs", type=int, default=10, help="Number of epochs to train with explanation loss"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--lexpl-reg", type=float, default=10, help="Regularization weight for symbolic explanation loss"
    )
    parser.add_argument(
        "--l1-reg", type=float, default=10, help="Regularization weight for sparsity symbolic explanation loss"
    )
    parser.add_argument(
        "--logic-batch-size", type=int, default=32, help="Batch size for forward reasoning"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size to train with"
    )
    parser.add_argument(
        "--reflect-trainset", type=bool, default=False, help="Use val or train loader for reflection"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of threads for data loader"
    )
    parser.add_argument(
        "--topk", type=int, default=5, help="How many maximally probable clauses should be extracted from the "
                                            "explanations for rrr feedback? "
                                            "NOTE: if it is 0 then the topk will start with as many as there arel "
                                            "lsx iters and constantly reduce by one per iteration"
    )
    parser.add_argument(
        "--unconfound-n-samples-per-class", type=int, default=0, help="Number of samples per class for unconfounded "
                                                                       "validation set. Used in case mode is unconfound"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run on CPU instead of GPU (not recommended)",
    )
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs")
    parser.add_argument(
        "--perc", type=float, default=1., help="How much percentage of the training data to use, between 0. and 1.."
    )
    parser.add_argument(
        "--prop-thresh", type=float, default=0.25, help="threshold (between 0 and 1) for considering symbolic "
                                                        "attribution value as relevant, i.e. for proposing logical "
                                                        "rule from explanation"
    )


    parser.add_argument("--data-dir", type=str, help="Directory to data")
    parser.add_argument("--fp-ckpt", type=str, default=None, help="checkpoint filepath")
    parser.add_argument("--fp-ckpt-vanilla", type=str, default=None, help="checkpoint filepath for vanilla model "
                                                                          "(e.g. for posthoc evaluations)")

    # Slot attention params
    parser.add_argument('--n-slots', default=10, type=int,
                        help='number of slots for slot attention module')
    parser.add_argument('--n-iters-slot-att', default=3, type=int,
                        help='number of iterations in slot attention module')
    parser.add_argument('--n-attr', default=18, type=int,
                        help='number of attributes per object')

    # NSFR args
    parser.add_argument("--m", type=int, default=1, help="The size of the logic program.")
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='Smooth parameter in the softor function')


    args = parser.parse_args()

    # hard set !!!!!!!!!!!!!!!!!!!!!!!!!
    args.n_heads = 4
    args.set_transf_hidden = 128

    args.dataset_type = 'clevr'

    assert args.data_dir.endswith(os.path.sep)
    args.conf_version = args.data_dir.split(os.path.sep)[-2]
    args.name = args.name + f"-{args.conf_version}" + f"-rnum{args.num}-{args.perc}"

    if args.dataset == "unconfound":
        args.name += "-unconfound"

    if args.no_cuda:
        args.device = 'cpu'
    else:
        args.device = 'cuda'

    set_seed(args.seed)

    args.init_topk = args.topk
    if args.topk == 0:
        args.topk = args.lsx_iters

    if args.mode != 'train':
        assert args.fp_ckpt is not None

    if args.mode == 'unconfound':
        assert args.unconfound_n_samples_per_class > 0

    return args


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
                                       class_expl_feedback_masks_inv):
    for dataloader in list_dataloaders:
        dataloader.dataset.class_expl_feedback_masks = class_expl_feedback_masks
        dataloader.dataset.class_expl_feedback_masks_inv = class_expl_feedback_masks_inv


def load_nn_model(args, fp_ckpt=None):
    # initialise NN model
    nn_model = nesycl_model.NeSyConceptLearner(n_classes=args.n_imgclasses, n_slots=args.n_slots,
                                          n_iters=args.n_iters_slot_att, n_attr=args.n_attr,
                                          n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden,
                                          category_ids=args.category_ids, device=args.device)
    # load pretrained state predictor
    log = torch.load("pretrained/slot-attention-clevr-state-3_final", map_location=torch.device(args.device))
    nn_model.img2state_net.load_state_dict(log['weights'], strict=True)
    print("Pretrained slot attention model loaded!")

    if fp_ckpt is not None:
        # load pretrained concept learner
        checkpoint = torch.load(fp_ckpt, map_location=torch.device('cpu'))
        nn_model.load_state_dict(checkpoint['nn_model'])
        print("Pretrained NeSy Concept Learner loaded!")

    nn_model = nn_model.to(args.device)

    # create optimizer
    params = list(
        [p for name, p in nn_model.named_parameters() if p.requires_grad and 'set_cls' in name])
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001)

    return nn_model, optimizer, scheduler


def write_clauses(clauses, writer, lsx_iter):
    for i in range(len(clauses)):
        writer.add_text(f"proposed clause {i}", clauses[i].__str__(), global_step=lsx_iter)
    with open(os.path.join(writer.log_dir, "clauses.csv"), "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(clauses)


def read_clauses(log_dir):
    with open(os.path.join(log_dir, "clauses.csv"), newline='') as f:
        reader = csv.reader(f)
        clauses = list(reader)
    return clauses


def save_ckpt(nn_model, logic_model, optimizer, val_loss, args, iter_idx, epoch, log_dir, best=False):
    save_dict = {
        "name": args.name,
        "nn_model": nn_model.state_dict(),
        "logic_model": logic_model.state_dict() if logic_model is not None else None,
        "optimizer": optimizer.state_dict(),
        "args": args,
    }
    if best:
        fp_save = os.path.join(log_dir, f"model_lsx_iter{iter_idx}_epoch{epoch}_bestvalloss_{val_loss:.4f}.pth")
    else:
        fp_save = os.path.join(log_dir, f"model_lsx_iter{iter_idx}_epoch{epoch}_finalloss_{val_loss:.4f}.pth")
    torch.save(save_dict, fp_save)
    return fp_save


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def get_dataloaders(args, mode='mix', pos_classid=0, onlyval=True):
    if mode == 'mix':
        dataset_train = data.CLEVR_HANS_EXPL(
            args.data_dir, "train_lsx", perc=args.perc, lexi=True, conf_vers=args.conf_version
        )
        print("training data loaded")
        dataset_val = data.CLEVR_HANS_EXPL(
            args.data_dir, "val_lsx", perc=1., lexi=True, conf_vers=args.conf_version
        )
        print("val data loaded")
        dataset_test = data.CLEVR_HANS_EXPL(
            args.data_dir, "val", lexi=True, conf_vers=args.conf_version
        )
        print("test data loaded")

        args.n_imgclasses = dataset_train.n_classes
        args.class_weights = torch.ones(args.n_imgclasses ) /args.n_imgclasses
        args.classes = np.arange(args.n_imgclasses)
        args.category_ids = dataset_train.category_ids

        train_loader = data.get_loader(
            dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
        )
        test_loader = data.get_loader(
            dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
        )
        val_loader = data.get_loader(
            dataset_val, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
        )
        return (train_loader, val_loader, test_loader)

    elif mode == 'pos':
        dataset_val_pos = data_bin.CLEVR_HANS_EXPL_bin_positive(
            args.data_dir, "val_lsx", perc=1., lexi=True, conf_vers=args.conf_version, pos_classid=pos_classid
        )
        val_loader = data.get_loader(
            dataset_val_pos, batch_size=args.logic_batch_size, num_workers=args.num_workers, shuffle=False,
        )

        if onlyval:
            return (val_loader)
        else:
            dataset_train_pos = data_bin.CLEVR_HANS_EXPL_bin_positive(
                args.data_dir, "train_lsx", perc=args.perc, lexi=True, conf_vers=args.conf_version, pos_classid=pos_classid,
            )
            dataset_test_pos = data_bin.CLEVR_HANS_EXPL_bin_positive(
                args.data_dir, "val", perc=args.perc, lexi=True, conf_vers=args.conf_version, pos_classid=pos_classid,
            )

            train_loader = data.get_loader(
                dataset_train_pos, batch_size=args.logic_batch_size, num_workers=args.num_workers, shuffle=True,
            )
            test_loader = data.get_loader(
                dataset_test_pos, batch_size=args.logic_batch_size, num_workers=args.num_workers, shuffle=False,
            )
            return (train_loader, val_loader, test_loader)


    elif mode == 'neg':
        dataset_val_neg = data_bin.CLEVR_HANS_EXPL_bin_negative(
            args.data_dir, "val_lsx", perc=1., lexi=True, conf_vers=args.conf_version, pos_classid=pos_classid,
        )
        val_loader = data.get_loader(
            dataset_val_neg, batch_size=args.logic_batch_size, num_workers=args.num_workers, shuffle=False,
        )

        if onlyval:
            return (val_loader)
        else:
            dataset_train_neg = data_bin.CLEVR_HANS_EXPL_bin_negative(
                args.data_dir, "train_lsx", perc=args.perc, lexi=True, conf_vers=args.conf_version, pos_classid=pos_classid,
            )
            dataset_test_neg = data_bin.CLEVR_HANS_EXPL_bin_negative(
                args.data_dir, "val", perc=args.perc, lexi=True, conf_vers=args.conf_version, pos_classid=pos_classid,
            )

            train_loader = data.get_loader(
                dataset_train_neg, batch_size=args.logic_batch_size, num_workers=args.num_workers, shuffle=True,
            )
            test_loader = data.get_loader(
                dataset_test_neg, batch_size=args.logic_batch_size, num_workers=args.num_workers, shuffle=False,
            )

            return (train_loader, val_loader, test_loader)


def norm_saliencies(saliencies):
    saliencies_norm = saliencies.clone()

    for i in range(saliencies.shape[0]):
        if len(torch.nonzero(saliencies[i], as_tuple=False)) == 0:
            saliencies_norm[i] = saliencies[i]
        else:
            saliencies_norm[i] = (saliencies[i] - torch.min(saliencies[i])) / \
                                 (torch.max(saliencies[i]) - torch.min(saliencies[i]))

    return saliencies_norm


def generate_intgrad_captum_table(net, x, obj_pres, labels, reasoner=True, device="cuda"):
    if x.requires_grad == False:
        x.requires_grad = True

    labels = labels.to(device)
    explainer = InputXGradient(net)
#     explainer = InputXGradient(net)
    saliencies = explainer.attribute(x, target=labels)
    # remove negative attributions
    saliencies[saliencies < 0] = 0.
    if reasoner:
        # normalise the explations and concatenate the object presence prediction back to explanation,
        # as nsfr requires this information
        saliencies_norm = torch.cat(
            (
            obj_pres,
            norm_saliencies(saliencies)
            ),
            dim=2
        )
    else:
        saliencies_norm = norm_saliencies(saliencies)
    return saliencies_norm


def generate_inpgrad_captum_table(net, x, obj_pres, labels, reasoner=True, device="cuda"):
    labels = labels.to(device)
    explainer = IntegratedGradients(net)
#     explainer = InputXGradient(net)
    saliencies = explainer.attribute(x, target=labels)
    # remove negative attributions
    saliencies[saliencies < 0] = 0.
    if reasoner:
        # normalise the explations and concatenate the object presence prediction back to explanation,
        # as nsfr requires this information
        saliencies_norm = torch.cat(
            (
            obj_pres,
            norm_saliencies(saliencies)
            ),
            dim=2
        )
    else:
        saliencies_norm = norm_saliencies(saliencies)
    return saliencies_norm


def generate_saliency_captum_table(net, x, obj_pres, labels, reasoner=True, device="cuda", norm=False):
    if x.requires_grad == False:
        x.requires_grad = True

    labels = labels.to(device)
    explainer = Saliency(net)
#     explainer = InputXGradient(net)
    saliencies = explainer.attribute(x, target=labels, abs=False)
    if norm:
        saliencies = norm_saliencies(saliencies)
    if reasoner:
        # normalise the explations and concatenate the object presence prediction back to explanation,
        # as nsfr requires this information
        saliencies = torch.cat(
            (
            obj_pres,
            saliencies
            ),
            dim=2
        )
    else:
        saliencies = saliencies
    return saliencies


def hungarian_matching(attrs, preds_attrs, verbose=0):
    """
    Receives unordered predicted set and orders this to match the nearest GT set.
    :param attrs:
    :param preds_attrs:
    :param verbose:
    :return:
    """
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


def write_expls(net, data_loader, tagname, epoch, writer, args):
    """
    Writes NeSy Concpet Learner explanations to tensorboard writer.
    """

    attr_labels = [
        'x', 'y', 'z',
        'Sphere', 'Cube', 'Cylinder',
        'Large', 'Small',
        'Rubber', 'Metal',
        'Cyan', 'Blue', 'Yellow', 'Purple', 'Red', 'Green', 'Gray', 'Brown'
    ]

    net.eval()

    for i, sample in enumerate(data_loader):
        # input is either a set or an image
        imgs, gt_attrs, gt_classes, img_ids, _, _, prop_expls, prop_expls_inv = map(lambda x: x.to(args.device), sample)
        gt_classes = gt_classes.long()

        # forward evaluation through the network
        output_cls, output_attrs, obj_pres = net.forward(imgs)
        _, preds = torch.max(output_cls, 1)

        # get explanations of set classifier
        model_symb_expls = generate_intgrad_captum_table(net.set_cls, output_attrs, obj_pres, preds, reasoner=False,
                                                         device=args.device)

        # convert sorting gt target set and gt table explanations to match the order of the predicted table
        gt_attrs, match_ids = hungarian_matching(output_attrs.to(args.device), gt_attrs)
        # table_expls = table_expls[:, match_ids][range(table_expls.shape[0]), range(table_expls.shape[0])]

        # get the ids of the two objects that receive the maximal importance, i.e. most important for the classification
        max_expl_obj_ids = model_symb_expls.max(dim=2)[0].topk(2)[1]

        # get attention masks
        attns = net.img2state_net.slot_attention.attn
        # reshape attention masks to 2D
        attns = attns.reshape((attns.shape[0], attns.shape[1], int(np.sqrt(attns.shape[2])),
                               int(np.sqrt(attns.shape[2]))))

        # concatenate the visual explanation of the top two objects that are most important for the classification
        img_saliencies = torch.zeros(attns.shape[0], attns.shape[2], attns.shape[3])
        for obj_id in range(max_expl_obj_ids.shape[1]):
            img_saliencies += attns[range(attns.shape[0]), obj_id, :, :].detach().cpu()

        # upscale img_saliencies to orig img shape
        img_saliencies = resize_tensor(img_saliencies.cpu(), imgs.shape[2], imgs.shape[2]).squeeze(dim=1).cpu()

        for img_id, (img, gt_attr, output_attr, model_symb_expl, img_expl,
                     true_label, pred_label, imgid, prop_expl, prop_expl_inv) in enumerate(zip(
                imgs, gt_attrs, output_attrs, model_symb_expls,
                img_saliencies, gt_classes, preds,
                img_ids, prop_expls, prop_expls_inv
        )):
            # unnormalize images
            img = img / 2. + 0.5  # Rescale to [0, 1].

            fig = create_expl_images(np.array(transforms.ToPILImage()(img.cpu()).convert("RGB")),
                                           output_attr.detach().cpu().numpy(),
                                           model_symb_expl.detach().cpu().numpy(),
                                           img_expl.detach().cpu().numpy(),
                                     prop_expl_inv.detach().cpu().numpy(),
                                           true_label, pred_label, attr_labels)
            writer.add_figure(f"{tagname}_{img_id}", fig, epoch)
            if img_id > 10:
                break

        break
