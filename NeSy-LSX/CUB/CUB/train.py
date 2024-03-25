"""
Train InceptionV3 Network using the CUB-200-2011 dataset
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
from torch.nn import functional as F

from analysis import Logger, AverageMeter, accuracy, binary_accuracy
from CUB.dataset import load_data, find_class_imbalance
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, MIN_LR, LR_DECAY_SIZE
from CUB.models import ModelXtoChat_ChatToY, ModelXtoY, ModelXtoC, ModelOracleCtoY, ModelXtoCtoY
import CUB.utils as utils

import rtpt

rtpt = rtpt.RTPT(name_initials="WS", experiment_name="concept bottleneck models", max_iterations=100)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_epoch_simple(model, optimizer, loader, loss_meter, acc_meter, criterion, is_training):
    """
    C -> Y: Predicting class labels using only attributes with MLP
    """
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        inputs, labels = data
        labels = labels.to(device)
        if isinstance(inputs, list):
            inputs = torch.stack(inputs).t()
        inputs = torch.flatten(inputs, start_dim=1).float().to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels, topk=(1,))
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc[0], inputs.size(0))

        if is_training:
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()
            optimizer.step()  # optimizer step to update parameters
    return loss_meter, acc_meter


def run_epoch(model, optimizer, loader, loss_meter, acc_meter, criterion, attr_criterion, args, is_training):
    """
    For the rest of the networks (X -> C, cotraining, simple finetune)
    """
    if is_training:
        model.train()
    else:
        model.eval()

    for _, data in enumerate(loader):
        if attr_criterion is None:
            inputs, labels = data
            attr_labels, attr_labels_var = None, None
        else:
            inputs, labels, attr_labels = data
            if args.n_attributes > 1:
                attr_labels = [i.long() for i in attr_labels]
                attr_labels = torch.stack(attr_labels).t()  #N x 312
            else:
                if isinstance(attr_labels, list):
                    attr_labels = attr_labels[0]
                attr_labels = attr_labels.unsqueeze(1)
            attr_labels_var = torch.autograd.Variable(attr_labels).float().to(device)

        inputs_var = torch.autograd.Variable(inputs).to(device)
        labels_var = torch.autograd.Variable(labels).to(device)

        if is_training and args.use_aux:
            outputs, aux_outputs = model(inputs_var)
            losses = []
            out_start = 0
            if not args.bottleneck:  # loss main is for the main task label (always the first output)
                loss_main = 1.0 * criterion(outputs[0], labels_var) + 0.4 * criterion(aux_outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if attr_criterion is not None and args.attr_loss_weight > 0:  # X -> C, cotraining, end2end
                for i in range(len(attr_criterion)):
                    losses.append(args.attr_loss_weight * (
                                1.0 * attr_criterion[i](outputs[i + out_start].squeeze(),
                                                        attr_labels_var[:, i])
                                + 0.4 * attr_criterion[i](aux_outputs[i + out_start].squeeze(),
                                                          attr_labels_var[:, i])))
        else:  # testing or no aux logits
            outputs = model(inputs_var)
            losses = []
            out_start = 0
            if not args.bottleneck:
                loss_main = criterion(outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if attr_criterion is not None and args.attr_loss_weight > 0:  # X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    losses.append(args.attr_loss_weight * attr_criterion[i](
                        outputs[i + out_start].squeeze(), attr_labels_var[:, i]))

        if args.bottleneck:  #attribute accuracy
            sigmoid_outputs = torch.sigmoid(torch.cat(outputs, dim=1))
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
            acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
        else:
            acc = accuracy(outputs[0], labels, topk=(1,))  # only care about class prediction accuracy
            acc_meter.update(acc[0], inputs.size(0))

        if attr_criterion is not None:
            if args.bottleneck:
                total_loss = sum(losses) / args.n_attributes
            else:  # cotraining, loss by class prediction and loss by attribute prediction have the same weight
                total_loss = losses[0] + sum(losses[1:])
                if args.normalize_loss:
                    total_loss = total_loss / (1 + args.attr_loss_weight * args.n_attributes)
        else:  # finetune
            total_loss = sum(losses)
        loss_meter.update(total_loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    return loss_meter, acc_meter


def train(model, args):
    rtpt.start()
    # Determine imbalance
    imbalance = None
    if args.use_attr and not args.no_img and args.weighted_loss:
        train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        if args.weighted_loss:
            imbalance = find_class_imbalance(train_data_path, True)
        else:
            imbalance = find_class_imbalance(train_data_path, False)

    if os.path.exists(args.log_dir):  # job restarted by cluster
        for f in os.listdir(args.log_dir):
            os.remove(os.path.join(args.log_dir, f))
    else:
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'))
    logger.write(str(args) + '\n')
    logger.write(str(imbalance) + '\n')
    logger.flush()

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    if args.use_attr and not args.no_img:
        attr_criterion = []  # separate criterion (loss function) for each attribute
        if args.weighted_loss:
            assert (imbalance is not None)
            for ratio in imbalance:
                attr_criterion.append(lambda src,target: F.binary_cross_entropy_with_logits(src, target=target, weight=torch.tensor([ratio], device=device, dtype=torch.float)))
        else:
            for i in range(args.n_attributes):
                attr_criterion.append(torch.nn.CrossEntropyLoss())
    else:
        attr_criterion = None

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
    stop_epoch = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    print("Stop epoch: ", stop_epoch)

    train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    logger.write('train data path: %s\n' % train_data_path)

    if args.ckpt:  # retraining
        train_loader = load_data([train_data_path, val_data_path], args.use_attr, args.no_img, args.batch_size,
                                 image_dir=args.image_dir, n_class=args.n_imgclasses)
        val_loader = None
    else:
        train_loader = load_data([train_data_path], args.use_attr, args.no_img, args.batch_size,
                                 image_dir=args.image_dir, n_class=args.n_imgclasses)
        val_loader = load_data([val_data_path], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
                               n_class=args.n_imgclasses)

    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(0, args.epochs):
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        if args.no_img:
            train_loss_meter, train_acc_meter = run_epoch_simple(model, optimizer, train_loader, train_loss_meter,
                                                                 train_acc_meter, criterion, is_training=True)
        else:
            train_loss_meter, train_acc_meter = run_epoch(model, optimizer, train_loader, train_loss_meter,
                                                          train_acc_meter, criterion, attr_criterion, args,
                                                          is_training=True)

        if args.ckpt:  # retraining on train and val set
            val_loss_meter = train_loss_meter
            val_acc_meter = train_acc_meter
        else:          # evaluate on val set
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()

            with torch.no_grad():
                if args.no_img:
                    val_loss_meter, val_acc_meter = run_epoch_simple(model, optimizer, val_loader, val_loss_meter,
                                                                     val_acc_meter, criterion, is_training=False)
                else:
                    val_loss_meter, val_acc_meter = run_epoch(model, optimizer, val_loader, val_loss_meter,
                                                              val_acc_meter, criterion, attr_criterion, args,
                                                              is_training=False)


        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            logger.write('New model best model at epoch %d\n' % epoch)
            torch.save(model, os.path.join(args.log_dir, 'best_model_%d.pth' % args.seed))

        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg

        logger.write('Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\t'
                     'Val loss: %.4f\tVal acc: %.4f\t'
                     'Best val epoch: %d\n'
                     % (epoch, train_loss_avg, train_acc_meter.avg, val_loss_avg, val_acc_meter.avg, best_val_epoch))
        logger.flush()

        if epoch <= stop_epoch:
            scheduler.step()  # scheduler step to update lr at the end of epoch
            rtpt.step(f'epoch:{epoch}')
        # inspect lr
        if epoch % 10 == 0:
            print('Current lr:', scheduler.get_last_lr())

    torch.save(model, os.path.join(args.log_dir, 'last_model_%d.pth' % args.seed))


def train_X_to_C(args):
    model = ModelXtoC(pretrained=args.pretrained, num_classes=args.n_imgclasses, use_aux=args.use_aux,
                      n_attributes=args.n_attributes)
    train(model, args)


def train_oracle_C_to_y_and_test_on_Chat(args):
    model = ModelOracleCtoY(n_attributes=args.n_attributes, num_classes=args.n_imgclasses)
    train(model, args)


def train_Chat_to_y_and_test_on_Chat(args):
    model = ModelXtoChat_ChatToY(n_attributes=args.n_attributes, num_classes=args.n_imgclasses)
    train(model, args)


def train_X_to_C_to_y(args):
    model = ModelXtoCtoY(pretrained=args.pretrained, num_classes=args.n_imgclasses, use_aux=args.use_aux,
                         n_attributes=args.n_attributes, use_sigmoid=args.use_sigmoid)
    train(model, args)


def train_X_to_y(args):
    model = ModelXtoY(pretrained=args.pretrained, num_classes=args.n_imgclasses, use_aux=args.use_aux)
    train(model, args)


def parse_arguments(experiment):
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='CUB Training')
    parser.add_argument('dataset', type=str, help='Name of the dataset.')
    parser.add_argument('exp', type=str,
                        choices=['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                                 'Standard', 'Multitask', 'Joint', 'Probe',
                                 'TTI', 'Robustness', 'HyperparameterSearch', 'SEL', 'faithfulness', 'sim_clf', 'sim'],
                        help='Name of experiment to run.')
    parser.add_argument('-seed', required=True, type=int, help='Numpy and torch seed.')

    parser.add_argument('-log_dir', default=None, help='where the trained model is saved')
    parser.add_argument('-batch_size', '-b', type=int, help='mini-batch size')
    parser.add_argument('-epochs', '-e', type=int, help='epochs for training process')
    parser.add_argument('-save_step', default=1000, type=int, help='number of epochs to save model')
    parser.add_argument('-lr', type=float, help="learning rate")
    parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
    parser.add_argument('-pretrained', '-p', action='store_true',
                        help='whether to load pretrained model & just fine-tune')
    parser.add_argument('-use_aux', action='store_true', help='whether to use aux logits')
    parser.add_argument('-use_attr', action='store_true',
                        help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)')
    parser.add_argument('-attr_loss_weight', default=1.0, type=float,
                        help='weight for loss by predicting attributes')
    parser.add_argument('-no_img', action='store_true',
                        help='if included, only use attributes (and not raw imgs) for class prediction')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels',
                        action='store_true')
    parser.add_argument('-weighted_loss', default='', action='store_true',
                        help='Whether to use weighted loss for single attribute or multiple ones')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES,
                        help='whether to apply bottlenecks to only a few attributes')
    parser.add_argument('-n_imgclasses', type=int, default=N_CLASSES,
                        help='whether to apply bottlenecks to only a few attributes')
    parser.add_argument('-data_dir', default='official_datasets', help='directory to the training data')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-end2end', action='store_true',
                        help='Whether to train X -> C -> Y end to end. Train cmd is the same as cotraining + this arg')
    parser.add_argument('-optimizer', default='SGD',
                        help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
    parser.add_argument('-ckpt', action='store_true', help='For retraining on both train + val set')
    parser.add_argument('-scheduler_step', type=int, default=1000,
                        help='Number of steps before decaying current learning rate by half')
    parser.add_argument('-normalize_loss', action='store_true',
                        help='Whether to normalize loss by taking attr_loss_weight into account')
    parser.add_argument('-use_relu', action='store_true',
                        help='Whether to include relu activation before using attributes to predict Y. '
                             'For end2end & bottleneck model')
    parser.add_argument('-use_sigmoid', action='store_true',
                        help='Whether to include sigmoid activation before using attributes to predict Y. '
                             'For end2end & bottleneck model')
    # --------------------------------------------------#
    # SEL args
    parser.add_argument(
        "-num", type=int, default=1, help="run number"
    )
    parser.add_argument(
        '-no_cuda', action='store_true'
    )
    parser.add_argument(
        "-topk", type=int, default=5, help="How many maximally probable clauses should be extracted from the "
                                           "explanations for rrr feedback? "
                                           "NOTE: if it is 0 then the topk will start with as many as there arel "
                                           "srl iters and constantly reduce by one per iteration"
    )
    parser.add_argument(
        "-expl-epochs", type=int, default=10, help="Number of epochs to train with explanation loss"
    )
    parser.add_argument(
        "-lexpl-reg", type=float, default=10, help="Regularization weight for symbolic explanation loss"
    )
    parser.add_argument(
        "-logic-batch-size", type=int, default=32, help="Batch size for forward reasoning"
    )
    # parser.add_argument(
    #     "-n_class", type=int, default=None, help="Number of classes to look at"
    # )
    parser.add_argument(
        "-srl-iters", type=int, default=1, help="Number of self-reflective learning iterations"
    )
    parser.add_argument("--fp-ckpt", type=str, default=None, help="checkpoint filepath")
    parser.add_argument(
        "-perc", type=float, default=1., help="How much percentage of the training data to use, between 0. and 1.."
    )
    parser.add_argument(
        "-prop-thresh", type=float, default=0.25, help="threshold (between 0 and 1) for considering symbolic "
                                                       "attribution value as relevant, i.e. for proposing logical "
                                                       "rule from explanation"
    )
    parser.add_argument(
        "-prop-min-thresh", type=float, default=0.25,
        help="percentage (between 0 and 1) of saliencies to consider, e.g. prop_min_thesh of the size of saliencies "
             "should be considered. Note in the end the dynamic prop_thresh is chosen from "
             "max(prop_threshold, torch.topk(saliences, k=int(prop_min_threshold * len(attr_saliencies)))[0][-1])"
    )
    parser.add_argument(
        "-max_n_atoms_per_clause", type=int, default=0,
        help="Maximal number of attributes/propositions per clause."
    )
    parser.add_argument(
        "-min_n_atoms_per_clause", type=int, default=0,
        help="How much lower than max_n_atoms_per_clause to go per clause."
    )
    # NSFR args
    parser.add_argument(
        "-m", type=int, default=1, help="The size of the logic program."
    )
    parser.add_argument(
        '-gamma', default=0.01, type=float,
        help='Smooth parameter in the softor function'
    )
    # parser.add_argument(
    #     "-fp_cat_attr_dict", type=str, default=None,
    #     help="filepath to dictionary containing category to attribute mapping"
    # )
    parser.add_argument(
        "-fp_ckpt", type=str, default=None,
        help="filepath to pretrained model"
    )
    parser.add_argument(
        "-fp_ckpt_sim", type=str, default=None,
        help="filepath to pretrained model for similarity measures"
    )
    # --------------------------------------------------#

    args = parser.parse_args()
    args.device = torch.device('cuda')
    if args.no_cuda or not torch.cuda.is_available():
        args.device = torch.device('cpu')

    utils.set_seed(args.seed)

    args.init_topk = args.topk

    args.fp_cat_attr_dict = 'NSFRAlpha/data/lang/cub/category_attributes_dict.pkl'

    args.dataset_type = 'cub'

    args.log_dir = os.path.join(args.log_dir.split(os.path.sep + 'outputs')[0], 'outputs')

    return (args,)
