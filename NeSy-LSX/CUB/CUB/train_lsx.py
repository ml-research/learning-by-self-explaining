"""
Train InceptionV3 Network using the CUB-200-2011 dataset
"""
import os
import sys
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import numpy as np
from analysis import Logger, AverageMeter, accuracy, binary_accuracy

from CUB.dataset import load_data, find_class_imbalance, get_loaders
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE
from CUB.models import ModelXtoChat_ChatToY
import LSX_utils.utils_reflect as reflect_utils
import LSX_utils.utils as expl_utils
from LSX_utils.reflect import reflect
from LSX_utils.rule_parsing import gen_masks_from_list_of_rules
from LSX_utils.xil_losses import rrr_loss_function, hint_loss_function, rrr_loss_function_captum
from LSX_utils.utils_faithfulness import comp_faithfulness
from LSX_utils.expl_sim_clf import comp_sim_clf
from LSX_utils.expl_sim import compute_intra_inter

from rtpt import RTPT


def load_nn_model(args, fp_ckpt=None):
    model = ModelXtoChat_ChatToY(n_attributes=args.n_attributes, num_classes=args.n_imgclasses)
    if fp_ckpt:
        model.load_state_dict(torch.load(fp_ckpt, map_location=torch.device(args.device)).state_dict())
        print(f"\nModel loaded from {fp_ckpt}\n")

    model.to(args.device)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                                        weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    return model, optimizer, scheduler


def run_epoch_simple(model, optimizer, loader, loss_meter, acc_meter, criterion, args, is_training,
                     rrr=False, hint=False):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """
    if is_training:
        model.train()
    else:
        model.eval()

    expl_reg = False
    if args.lexpl_reg > 0:
        expl_reg = True

    for _, data in enumerate(loader):
        if expl_reg:
            inputs, labels, prop_expl, prop_expl_inv = data
        else:
            inputs, labels = data
        labels = labels.to(args.device)
        if isinstance(inputs, list):
            inputs = torch.stack(inputs).t() # .float()
        inputs = torch.flatten(inputs, start_dim=1).float().to(args.device)

        outputs = model(inputs)
        pred_loss = criterion(outputs, labels) #_var)

        expl_loss = torch.zeros(1, requires_grad=True, device=args.device)
        if expl_reg:
            # get explanations of model
            model_symb_expl = expl_utils.generate_intgrad_captum_table(
                model, inputs, labels, device=args.device
            ).unsqueeze(dim=1)
            if rrr:
                expl_loss = rrr_loss_function_captum(A=prop_expl_inv, model_grads=model_symb_expl)
            elif hint:
                # perform hint-like loss
                expl_loss = hint_loss_function(model_symb_expl, prop_expl)

        loss = pred_loss + args.lexpl_reg * expl_loss

        acc = accuracy(outputs, labels, topk=(1,))
        loss_meter.update(loss.item(), pred_loss.item(), expl_loss.item(), n=inputs.size(0))
        acc_meter.update(acc[0], inputs.size(0))

        if is_training:
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()
            optimizer.step()  # optimizer step to update parameters
    return loss_meter, acc_meter


def train(model, loaders, optimizer, scheduler, args, logger, rtpt, lsx_iter, rrr=False, hint=False):

    train_loader, train_val_loader, val_loader, test_loader = loaders

    # create criteria
    criterion = torch.nn.CrossEntropyLoss()

    # for determinism --> makes it slower
    torch.backends.cudnn.benchmark = True

    best_val_acc = 0
    best_val_epoch = 0
    epochs = args.epochs
    if lsx_iter % 0.5 == 0 and lsx_iter > 0.0:
        epochs = args.expl_epochs

    for epoch in range(epochs):

        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()

        train_loss_meter, train_acc_meter = run_epoch_simple(model, optimizer, train_loader, train_loss_meter,
                                                             train_acc_meter, criterion, args, is_training=True,
                                                             rrr=rrr, hint=hint)

        if not args.ckpt:  # evaluate on val set
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()

            with torch.no_grad():
                val_loss_meter, val_acc_meter = run_epoch_simple(model, optimizer, val_loader, val_loss_meter,
                                                                 val_acc_meter, criterion, args, is_training=False)
        else:  # retraining
            val_loss_meter = train_loss_meter
            val_acc_meter = train_acc_meter

        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            logger.write('New model best model at epoch %d\n' % epoch)
            torch.save(model, os.path.join(args.log_dir, f"best_model_{args.seed}_{args.lsx_iter}.pth"))

        train_loss_avg = train_loss_meter.avg
        train_loss_pred_avg = train_loss_meter.avg_pred
        train_loss_expl_avg = train_loss_meter.avg_expl
        val_loss_avg = val_loss_meter.avg

        logger.write('Epoch [%d]:\tTrain loss: %.4f\tTrain loss Pred: %.4f\tTrain loss Expl: %.4f\tTrain accuracy: %.4f\t'
                     'Val loss: %.4f\tVal acc: %.4f\t'
                     'Best val epoch: %d\n'
                     % (epoch, train_loss_avg, train_loss_pred_avg, train_loss_expl_avg, train_acc_meter.avg,
                        val_loss_avg, val_acc_meter.avg, best_val_epoch))
        logger.flush()

        scheduler.step()  # scheduler step to update lr at the end of epoch
        if epoch % 10 == 0:
            print('Current lr:', scheduler.get_last_lr())


def self_reflect(args):

    model, loaders, optimizer, scheduler, logger = get_model_data_optimizer_etc(args)

    train_loader, train_val_loader, val_loader, test_loader = loaders

    logger.write(f"\n{train_loader.dataset.__len__()} samples in train, "
                 f"{val_loader.dataset.__len__()} in validation and "
                 f"{test_loader.dataset.__len__()} in test split\n")

    # Create and start RTPT object
    rtpt = RTPT(name_initials='WS', experiment_name=f"CUB SelfReflectiveCoder",
                max_iterations=np.max((1, args.lsx_iters)))
    rtpt.start()

    RRR = False
    HINT = False
    # ---- Step 1: pretrain NN without reasoner feedback ----
    logger.write("\nStep 1: Pretrain NN #------------------------------------------------------------------------------#\n")
    if args.fp_ckpt is None:
        args.lexpl_reg = 0.
        args.lsx_iter = 0
        _ = train(model, loaders, optimizer, scheduler, args, logger, rtpt,
                         args.lsx_iter, rrr=RRR, hint=HINT)
        torch.save(model, os.path.join(args.log_dir, f"last_model_{args.seed}_{args.lsx_iter}.pth"))

    # apply rrr loss, if it is the last iteration apply hint so only one rule is true
    RRR = True
    for lsx_iter in range(args.lsx_iters):
        """
        single loop of SRL
        """
        args.lsx_iter = lsx_iter
        # if the number of positive rules should reduce over lsx iterations, reduce current value by one
        if args.init_topk == 0:
            args.topk = np.max(1, args.topk-1)

        # if it is the last lsx iteration only extract the max probable clause per class and set hint loss to true
        args.lexpl_reg = 10.
        if args.lsx_iter == (args.lsx_iters - 1):
            args.topk = 1
            RRR = False
            HINT = True
            args.lexpl_reg = 100.
            print("\nOptimizing via Hint loss!")

        # # ---- Step 2: propositionalise knowledge rules from local explanations ----
        print("\nStep 2 : Propositionalising rules from attribute saliencies #--------------------------------------#")
        print("Using train set for reflection")
        reflect_utils.propositionalise(model, train_loader, args.lsx_iter, args)
        train_loader.dataset.sample_ids = train_loader.dataset.full_sample_ids
        del model

        # ---- Step 3: perform forward reasoning over proposed global rules to find best explaining rule ----
        print("\nStep 3 : Extract most probable clause(s) via forward reasoning #-----------------------------------#")
        print("Using train_val_loader for NSFR scoring")
        orig_batch_size = args.batch_size
        args.batch_size = 4
        _, train_val_loader, _, _ = get_loaders(BASE_DIR, args)
        _, clauses = reflect(args.lsx_iter, rr=True, topk=args.topk, args=args, loader=train_val_loader, verbose=0)
        expl_utils.write_clauses(clauses, args.lsx_iter, args.log_dir)
        args.batch_size = orig_batch_size
        train_loader, train_val_loader, val_loader, test_loader = loaders

        # ---- Step 4: retrain with explanation feedback ----
        print("\nStep 4 : Retrain NN via expl feedback #------------------------------------------------------------#")
        # transform clauses into feedback masks per class for right reason loss
        clauses = expl_utils.read_clauses(args.log_dir)
        class_expl_feedback_masks, class_expl_feedback_masks_inv = gen_masks_from_list_of_rules(
            clauses, max_objs=1, rrr=RRR
        )
        # pass class feedback back to train and val dataloader
        expl_utils.add_feedback_to_datasets([train_loader, val_loader], class_expl_feedback_masks,
                                            class_expl_feedback_masks_inv, device=args.device)
        # Currently I am starting from scratch instead of last trained ckpt
        model, optimizer, scheduler = load_nn_model(args, fp_ckpt=None)

        args.lsx_iter+=.5 # to differentiate the training subloops
        _ = train(model, loaders, optimizer, scheduler, args, logger, rtpt,
                         args.lsx_iter, rrr=RRR, hint=HINT)
        torch.save(model, os.path.join(args.log_dir, f"last_model_{args.seed}_{args.lsx_iter}.pth"))


def get_model_data_optimizer_etc(args):
    model, optimizer, scheduler = load_nn_model(args, fp_ckpt=args.fp_ckpt)

    # Determine imbalance
    imbalance = None
    if args.use_attr and not args.no_img and args.weighted_loss:
        train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        if args.weighted_loss == 'multiple':
            imbalance = find_class_imbalance(train_data_path, True)
        else:
            imbalance = find_class_imbalance(train_data_path, False)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, f"log.txt"))
    logger.write(str(args) + '\n')
    logger.write(str(imbalance) + '\n')
    logger.flush()

    stop_epoch = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    print("Stop epoch: ", stop_epoch)

    loaders = get_loaders(BASE_DIR, args)

    return model, loaders, optimizer, scheduler, logger


def train_Chat_to_y_and_test_on_Chat_LSX(args):
    args.no_img = True
    self_reflect(args)


def faithfulness(args):
    args.no_img = True
    model, _, _ = load_nn_model(args, fp_ckpt=args.fp_ckpt)
    model.eval()

    loaders = get_loaders(BASE_DIR, args)
    comp, suff = comp_faithfulness(model, loaders, args)

    print(f"Faithfulness of model {args.fp_ckpt} on test set: COMP: {comp} SUFF: {suff}")


def sim(args):
    args.no_img = True
    model, _, _ = load_nn_model(args, fp_ckpt=args.fp_ckpt)
    model.eval()
    # the model to store the encodings from
    model_sim, _, _ = load_nn_model(args, fp_ckpt=args.fp_ckpt_sim)
    model_sim.eval()

    _, _, _, test_loader = get_loaders(BASE_DIR, args)
    _, mean_avg_inter_mean_intra, _, _ = compute_intra_inter(args, model, model_sim, test_loader)

    print(f"Sim of model {args.fp_ckpt} on test set: {mean_avg_inter_mean_intra}")


def sim_clf(args):
    args.no_img = True
    model, _, _ = load_nn_model(args, fp_ckpt=args.fp_ckpt)
    model.eval()

    _, _, _, test_loader = get_loaders(BASE_DIR, args)
    sim_clf_acc = comp_sim_clf(model, test_loader, args)

    print(f"RR acc: {sim_clf_acc}")