import sys
sys.path.append('NSFRAlpha/src/')
import matplotlib
matplotlib.use("Agg")
import os
import torch
import numpy as np
import glob
from sklearn import metrics
from tqdm import tqdm
from rtpt import RTPT

import NeSyConceptLearner.src.model_for_nsfr as nesycl_model
import NeSyConceptLearner.src.utils as nesycl_utils
import utils as utils
import utils_reflect as reflect_utils
import utils_faithfulness as utils_faithfulness
import utils_unconfound as utils_unconfound
from reflect import reflect
from NSFRAlpha.src.logic_utils import get_lang, get_searched_clauses
from NSFRAlpha.src.nsfr_utils import denormalize_kandinsky, get_data_loader, get_data_pos_loader, get_data_neg_loader, \
    get_prob, get_nsfr_model, get_expl_nsfr_model, update_initial_clauses
from rule_parsing import gen_masks_from_list_of_rules
from xil_losses import rrr_loss_function, hint_loss_function, rrr_loss_function_captum

os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"
torch.set_num_threads(6)

# -----------------------------------------
# - Define basic and data related methods -
# -----------------------------------------
def get_confusion_from_ckpt(net, test_loader, criterion, args, datasplit, writer=None):

    true, pred, true_wrong, pred_wrong = run_test_final(net, test_loader, criterion, writer, args, datasplit)
    precision, recall, accuracy, f1_score = nesycl_utils.performance_matrix(true, pred)

    # Generate Confusion Matrix
    if writer is not None:
        nesycl_utils.plot_confusion_matrix(true, pred, normalize=True, classes=args.classes,
                              sFigName=os.path.join(writer.log_dir, 'Confusion_matrix_normalize_{}.pdf'.format(
                                  datasplit))
                              )
        nesycl_utils.plot_confusion_matrix(true, pred, normalize=False, classes=args.classes,
                              sFigName=os.path.join(writer.log_dir, 'Confusion_matrix_{}.pdf'.format(datasplit)))
    else:
        nesycl_utils.plot_confusion_matrix(true, pred, normalize=True, classes=args.classes,
                              sFigName=os.path.join(os.path.sep.join(args.fp_ckpt.split(os.path.sep)[:-1]),
                                                    'Confusion_matrix_normalize_{}.pdf'.format(datasplit)))
        nesycl_utils.plot_confusion_matrix(true, pred, normalize=False, classes=args.classes,
                              sFigName=os.path.join(os.path.sep.join(args.fp_ckpt.split(os.path.sep)[:-1]),
                                                    'Confusion_matrix_{}.pdf'.format(datasplit)))
    return accuracy

# -----------------------------------------
# - Define Train/Test/Validation methods -
# -----------------------------------------
def run_test_final(net, loader, criterion, writer, args, datasplit):
    net.eval()

    running_corrects = 0
    running_loss=0
    pred_wrong = []
    true_wrong = []
    preds_all = []
    labels_all = []
    with torch.no_grad():

        for i, sample in enumerate(tqdm(loader)):
            # input is either a set or an image
            imgs, gt_attr, gt_classes, _, _, table_expl = map(lambda x: x.to(args.device), sample)
            gt_classes = gt_classes.long()

            # forward evaluation through the network
            output_cls, output_attr = net(imgs)

            # class prediction
            _, preds = torch.max(output_cls, 1)

            labels_all.extend(gt_classes.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())

            running_corrects = running_corrects + torch.sum(preds == gt_classes)
            loss = criterion(output_cls, gt_classes)
            running_loss += loss.item()
            preds = preds.cpu().numpy()
            target = gt_classes.cpu().numpy()
            preds = np.reshape(preds, (len(preds), 1))
            target = np.reshape(target, (len(preds), 1))

            for i in range(len(preds)):
                if (preds[i] != target[i]):
                    pred_wrong.append(preds[i])
                    true_wrong.append(target[i])

        bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

        if writer is not None:
            writer.add_scalar(f"Loss/{datasplit}_loss", running_loss / len(loader), 0)
            writer.add_scalar(f"Acc/{datasplit}_bal_acc", bal_acc, 0)

        return labels_all, preds_all, true_wrong, pred_wrong


def run(net, loader, optimizer, criterion, split, writer, args, train=False, plot=False, epoch=0, lsx_iter=0):
    if train:
        net.img2state_net.eval()
        net.set_cls.train()
        torch.set_grad_enabled(True)
    else:
        net.eval()
        torch.set_grad_enabled(False)

    iters_per_epoch = len(loader)
    loader = tqdm(
        loader,
        ncols=0,
        desc=f"{lsx_iter} {'train' if train else 'val'} E{epoch:02d}",
    )
    running_loss = 0
    preds_all = []
    labels_all = []
    for i, sample in enumerate(loader, start=epoch * iters_per_epoch):
        # input is either a set or an image
        imgs, gt_attr, gt_classes, _, _, _, _, _ = map(lambda x: x.to(args.device), sample)
        gt_classes = gt_classes.long()

        # forward evaluation through the network
        output_cls, output_attr, _ = net(imgs)

        # class prediction
        _, preds = torch.max(output_cls, 1)

        loss = criterion(output_cls, gt_classes)

        # Outer optim step
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        labels_all.extend(gt_classes.cpu().numpy())
        preds_all.extend(preds.cpu().numpy())

        # Plot predictions in Tensorboard
        if plot and not(i % iters_per_epoch):
            nesycl_utils.write_expls(net, loader, f"{lsx_iter}_Expl/{split}", epoch, writer)

    bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

    if writer is not None:
        writer.add_scalar(f"{lsx_iter}_Loss/{split}_loss", running_loss / len(loader), epoch)
        writer.add_scalar(f"{lsx_iter}_Acc/{split}_bal_acc", bal_acc, epoch)

    print("Epoch: {}/{}.. ".format(epoch, args.epochs),
          "{} Loss: {:.3f}.. ".format(split, running_loss / len(loader)),
          "{} Accuracy: {:.3f}.. ".format(split, bal_acc),
          )

    return running_loss / len(loader), bal_acc


def run_train_step(nn_model, loader, optimizer, criterion,
                    split, writer, args, rrr=False, hint=False, plot=False, epoch=0, lsx_iter=0):

    assert not(rrr and hint)

    if split == 'train':
        nn_model.img2state_net.eval()
        nn_model.set_cls.train()
        torch.set_grad_enabled(True)
        train = True
    else:
        nn_model.eval()
        torch.set_grad_enabled(False)
        train = False

    # Plot initial predictions in Tensorboard
    if plot and epoch == 0:
        utils.write_expls(nn_model, loader, f"{lsx_iter}_Expl/{split}", -1, writer, args)

    running_loss = 0
    pred_running_loss = 0
    expl_running_loss = 0
    preds_all = []
    labels_all = []

    iters_per_epoch = len(loader)
    loader = tqdm(
        loader,
        ncols=0,
        desc=f"{lsx_iter} {'train' if train else 'val'} E{epoch:02d}",
    )

    expl_reg = False
    if args.lexpl_reg > 0:
        expl_reg = True

    for i, sample in enumerate(loader, start=epoch * iters_per_epoch):

        # input is either a set or an image
        imgs, _, gt_classes, _, _, _, prop_expl, prop_expl_inv = map(lambda x: x.to(args.device), sample)
        gt_classes = gt_classes.long()

        # forward evaluation through the network
        output_cls, output_attr, obj_pres = nn_model.forward(imgs)
        _, preds = torch.max(output_cls, 1)

        # prediction loss
        pred_loss = criterion(output_cls, gt_classes.long())

        expl_loss = torch.zeros(1, requires_grad=True, device=args.device)
        # if logical feedback should be added on explanations
        if expl_reg:
            if rrr:
                model_symb_expl = utils.generate_intgrad_captum_table(nn_model.set_cls, output_attr, obj_pres,
                                                                      gt_classes, device=args.device,
                                                                      reasoner=False)
                # compute right reason loss
                expl_loss = rrr_loss_function_captum(A=prop_expl_inv, model_grads=model_symb_expl)
            elif hint:
                # convert sorting gt target set and gt table explanations to match the order of the predicted table
                prop_expl, match_ids = utils.hungarian_matching(output_attr.to(args.device), prop_expl.float())

                # get explanations of set classifier
                model_symb_expl = utils.generate_intgrad_captum_table(nn_model.set_cls, output_attr, obj_pres,
                                                                      gt_classes, device=args.device,
                                                                      reasoner=False)
                # perform hint-like loss
                expl_loss = hint_loss_function(model_symb_expl, prop_expl)

        loss = pred_loss + args.lexpl_reg * expl_loss

        running_loss += loss.item()
        pred_running_loss += pred_loss.item()
        expl_running_loss += expl_loss.item()

        optimizer.zero_grad()
        if train:
            loss.backward()
            optimizer.step()

        labels_all.extend(gt_classes.detach().cpu().numpy())
        preds_all.extend(preds.detach().cpu().numpy())

        # Plot predictions in Tensorboard
        if plot and not(i % iters_per_epoch) and (epoch % 10) == 0:
            utils.write_expls(nn_model, loader, f"{lsx_iter}_Expl/{split}", epoch, writer, args)

        del imgs, gt_classes, output_cls, output_attr, obj_pres, prop_expl, prop_expl_inv

    bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

    writer.add_scalar(f"{lsx_iter}_Loss/{split}_loss", running_loss / len(loader), epoch)
    writer.add_scalar(f"{lsx_iter}_Loss/{split}_ra_loss", pred_running_loss / len(loader), epoch)
    writer.add_scalar(f"{lsx_iter}_Loss/{split}_expl_loss", expl_running_loss / len(loader), epoch)
    writer.add_scalar(f"{lsx_iter}_Acc/{split}_bal_acc", bal_acc, epoch)

    print("{} Loss: {:.4f}.. ".format(split, running_loss / len(loader)),
          "{} RA Loss: {:.4f}.. ".format(split, pred_running_loss / len(loader)),
          "{} Expl Loss: {:.4f}.. ".format(split, expl_running_loss / len(loader)),
          "{} Accuracy: {:.4f}.. ".format(split, bal_acc),
          )

    return running_loss / len(loader)


def train(nn_model, loaders, optimizer, scheduler, writer, args, lsx_iter, rrr=False, hint=False):

    train_loader, val_loader, test_loader = loaders

    # create criteria
    criterion = torch.nn.CrossEntropyLoss()

    # for determinism --> makes it slower
    torch.backends.cudnn.benchmark = True

    cur_best_val_loss = np.inf
    epochs = args.epochs
    if lsx_iter % 0.5 == 0 and lsx_iter > 0.0:
        epochs = args.expl_epochs

    cur_best_val_loss = np.inf
    for epoch in range(epochs):
        # train
        _ = run_train_step(nn_model, train_loader, optimizer, criterion,
                           split='train', args=args, writer=writer, plot=False, epoch=epoch,
                           lsx_iter=lsx_iter, rrr=rrr, hint=hint)
        scheduler.step()

        # validation
        val_loss = run_train_step(nn_model, val_loader, optimizer, criterion,
                                  split='val', args=args, writer=writer, plot=True, epoch=epoch,
                                  lsx_iter=lsx_iter, rrr=False, hint=False)
        if cur_best_val_loss > val_loss:
            print("Saving new best val loss model ...")
            if epoch > 0:
                # remove previous best model
                os.remove(glob.glob(os.path.join(args.log_dir, f"model_lsx_iter{args.lsx_iter}*bestvalloss*.pth"))[0])
            fp_ckpt_best = utils.save_ckpt(nn_model, None, optimizer, val_loss, args, args.lsx_iter, epoch,
                                           args.log_dir, best=True)
            cur_best_val_loss = val_loss

    fp_ckpt_best = utils.save_ckpt(nn_model, None, optimizer, val_loss, args, args.lsx_iter, epoch, args.log_dir)

    # test
    nn_model, optimizer, scheduler = utils.load_nn_model(args, fp_ckpt=fp_ckpt_best)
    _, _ = run(nn_model, test_loader, optimizer, criterion, split='test', args=args, writer=writer,
            train=False, plot=False, epoch=epoch, lsx_iter=lsx_iter)

    return val_loss


def self_reflect(nn_model, optimizer, scheduler, loaders, writer, args):

    train_loader, val_loader, _ = loaders

    print(f"\n{train_loader.dataset.__len__()} samples in train\n")

    # Create and start RTPT object
    rtpt = RTPT(name_initials='WS', experiment_name=f"SelfReflectiveCoder",
                max_iterations=np.max((1, args.lsx_iters)))
    rtpt.start()

    RRR = False
    HINT = False
    # ---- Step 1: pretrain NN without reasoner feedback ----
    print("\nStep 1: Pretrain NN #----------------------------------------------------------------------------------#")
    if args.fp_ckpt is None:
        args.lexpl_reg = 0.
        args.lsx_iter = 0
        _ = train(nn_model, loaders, optimizer, scheduler, writer, args,
                         args.lsx_iter, rrr=RRR, hint=HINT)

    # apply rrr loss, if it is the last iteration apply hint so only one rule is true
    RRR = True
    for lsx_iter in range(args.lsx_iters):
        """
        single loop of SRL
        """
        args.lsx_iter = lsx_iter
        args.lexpl_reg = 10.
        # if it is the last iteration apply hint so only one rule is true
        if args.lsx_iter == (args.lsx_iters - 1):
            args.topk = 1
            RRR = False
            HINT = True
            args.lexpl_reg = 100.
            print("\nOptimizing via Hint loss!")

        # # ---- Step 2: propositionalise knowledge rules from local explanations ----
        print("\nStep 2 : Propositionalising rules from attribute saliencies #--------------------------------------#")
        print("Using train set for reflection")
        reflect_utils.propositionalise(nn_model, train_loader, args.lsx_iter, args)
        del nn_model

        # ---- Step 3: perform forward reasoning over proposed global rules to find best explaining rule ----
        print("\nStep 3 : Extract most probable clause(s) via forward reasoning #-----------------------------------#")
        _, clauses = reflect(args.lsx_iter, rr=True, topk=args.topk, args=args, loader=train_loader)
        utils.write_clauses(clauses, writer, args.lsx_iter)
        clauses = utils.read_clauses(args.log_dir)

        # transform clauses into feedback masks per class for right reason loss
        class_expl_feedback_masks, class_expl_feedback_masks_inv = gen_masks_from_list_of_rules(
            clauses, max_objs=args.n_slots, rrr=RRR
        )
        # pass class feedback back to train and val dataloader
        utils.add_feedback_to_datasets([loaders[0], loaders[1]], class_expl_feedback_masks,
                                       class_expl_feedback_masks_inv)

        # ---- Step 4: retrain with explanation feedback ----
        print("\nStep 4 : Retrain NN via expl feedback #------------------------------------------------------------#")
        # TODO: currently I am starting from scratch instead of last trained ckpt
        nn_model, optimizer, scheduler = utils.load_nn_model(args, fp_ckpt=None)
        args.lsx_iter+=.5 # to differentiate the training subloops
        _ = train(nn_model, loaders, optimizer, scheduler, writer, args,
                         args.lsx_iter, rrr=RRR, hint=HINT)


def apply_feedback(loaders, writer, args):
    clauses = utils.read_clauses(args.log_dir)

    # transform clauses into feedback masks per class for right reason loss
    class_expl_feedback_masks, class_expl_feedback_masks_inv = gen_masks_from_list_of_rules(
        clauses, max_objs=args.n_slots, rrr=RRR
    )
    # pass class feedback back to train and val dataloader
    utils.add_feedback_to_datasets([loaders[0], loaders[1]], class_expl_feedback_masks,
                                   class_expl_feedback_masks_inv)

    # ---- Step 4: retrain with explanation feedback ----
    print("\nStep 4 : Retrain NN via user expl feedback #---------------------------------------------------------#")
    # TODO: currently I am starting from scratch instead of last trained ckpt
    nn_model, optimizer, scheduler = utils.load_nn_model(args, fp_ckpt=None)
    args.lsx_iter += .5  # to differentiate the training subloops
    _ = train(nn_model, loaders, optimizer, scheduler, writer, args,
              args.lsx_iter, rrr=RRR, hint=HINT)


def eval(nn_model, test_loader, optimizer, writer, args):
    # create criteria
    criterion = torch.nn.CrossEntropyLoss()

    # test
    _, bal_acc = run(nn_model, test_loader, optimizer, criterion, split='test', args=args, writer=writer,
            train=False, plot=False, epoch=-1, lsx_iter=-1)
    print(f"Test accuracy: {bal_acc}")


def main():
    args = utils.get_args()

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

    # tensorboard writer
    if args.mode in ['train', 'feedback']:
        writer = utils.create_writer(args)
        args.log_dir = writer.log_dir
        if args.mode == 'train':
            utils.save_args(args)
        writer.add_scalar(f"data_size/train", loaders[0].dataset.__len__(), 0)
        writer.add_scalar(f"data_size/val", loaders[1].dataset.__len__(), 0)
        writer.add_scalar(f"data_size/test", loaders[2].dataset.__len__(), 0)
    else:
        args.log_dir = os.path.sep.join(args.fp_ckpt.split(os.path.sep)[:-1])

    if args.fp_ckpt != None:
        print(f"Loading {args.fp_ckpt}")
        args.model_ckpt = args.fp_ckpt.split(args.log_dir+os.path.sep)[-1].split('.pth')[0]

    nn_model, optimizer, scheduler = utils.load_nn_model(args, fp_ckpt=args.fp_ckpt)

    if args.mode == 'train':
        self_reflect(nn_model, optimizer, scheduler, loaders, writer, args)
    elif args.mode == 'test':
        eval(nn_model, loaders[2], optimizer, None, args)
    elif args.mode == 'faithfulness':
        utils_faithfulness.comp_faithfulness(nn_model, loaders, args)
    elif args.mode == 'feedback':
        apply_feedback(loaders, writer, args)

if __name__ == "__main__":
    main()
