from typing import Tuple, List, Optional

import torch
import global_vars
import pickle
import os
import numpy as np
import utils as utils
import utils_faithfulness as utils_faithfulness
from critic import Critic
from learner import Learner
from visualization import ImageHandler
from net import SimpleConvNet


def run_experiments(overriding_args: Optional[List] = None):

    print("Setting up experiments...")
    args = utils.parse_args(overriding_args)
    utils.setup(args)
    print(global_vars.DEVICE)

    loaders = utils.load_data_from_args(args)
    test_batch_to_visualize = utils.get_one_batch_of_images(loaders.visualization)

    # start at the negative pretraining iterations, so the logging of joint training starts at step zero.
    if 'pretrain_and_joint' in args.training_mode:
        global_vars.global_step = -(args.n_iterations - args.joint_iterations)
    else:
        global_vars.global_step = 0

    # initialize learner object
    learner_model_fn, critic_model_fn = utils.get_model_fn(args)
    model_path_str = "" if args.logging_disabled else f"{global_vars.LOGGING.writer.log_dir}/{utils.config_string(args)}.pt"
    learner = Learner(learner_model_fn, critic_model_fn, loaders,
                          optimizer_type=args.optimizer,
                          test_batch_to_visualize=test_batch_to_visualize,
                          model_path=model_path_str,
                          explanation_mode=args.explanation_mode)

    if args.training_mode == "joint":
        utils.get_initial_expl_imgs(test_batch_to_visualize, learner, ImageHandler)
        print("Joint training together with simple joint loss...")
        init_loss, fin_loss = learner.train_from_args(args)
        print(f"initial/final loss:{init_loss:.3f}, {fin_loss:3f}")
        ImageHandler.add_gradient_images(test_batch=test_batch_to_visualize, learner=learner,
                                         additional_caption="2: after joint training")

    elif args.training_mode == "pretrain_and_joint":
        utils.get_initial_expl_imgs(test_batch_to_visualize, learner, ImageHandler)
        print("Pre-train the learner first and then perform joint training ...")
        init_loss_pretraining, final_loss_pretraining = learner.pretrain_from_args(args)
        print(f"initial/final loss (pretraining):{init_loss_pretraining:.3f}, {final_loss_pretraining:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, learner, additional_caption="1: after pretraining")

        init_loss, fin_loss = learner.train_from_args(args)
        print(f"initial/final loss (joint, after pretraining):{init_loss:.3f}, {fin_loss:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, learner,
                                         additional_caption="2: after joint training")
        # ImageHandler.add_input_images(test_batch_to_visualize[0])

    elif args.training_mode == "pretrain_and_joint_and_finetuning":
        utils.get_initial_expl_imgs(test_batch_to_visualize, learner, ImageHandler)
        print("Pre-train the learner first and then perform joint training ...")
        init_loss_pretraining, final_loss_pretraining = learner.pretrain_from_args(args)
        print(f"initial/final loss (pretraining):{init_loss_pretraining:.3f}, {final_loss_pretraining:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, learner, additional_caption="1: after pretraining")

        init_loss, fin_loss = learner.train_from_args(args)
        print(f"initial/final loss (joint, after pretraining):{init_loss:.3f}, {fin_loss:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, learner,
                                         additional_caption="2: after joint training")
        # ImageHandler.add_input_images(test_batch_to_visualize[0])

        init_loss, fin_loss = learner.finetune_on_explanations(args)
        print(f"initial/final loss (finetuning after joint):{init_loss:.3f}, {fin_loss:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, learner,
                                         additional_caption="3: after finetuning")
        # ImageHandler.add_input_images(test_batch_to_visualize[0])

    elif args.training_mode == 'finetuning':
        utils.get_initial_expl_imgs(test_batch_to_visualize, learner, ImageHandler)
        assert args.model_pt is not None
        log = torch.load(args.model_pt, map_location=torch.device(global_vars.DEVICE))
        learner.classifier.load_state_dict(log['model_state_dict'])
        print("Model loaded for finetuning")
        init_loss, fin_loss = learner.finetune_on_explanations(args)
        print(f"initial/final loss (finetuning after joint):{init_loss:.3f}, {fin_loss:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, learner,
                                         additional_caption="3: after finetuning")
        # ImageHandler.add_input_images(test_batch_to_visualize[0])

    elif args.training_mode == "pretrained":
        print("Pre-train the learner first and then perform joint training ...")
        learner.load_state("./models/pretrained_model.pt")
        ImageHandler.add_gradient_images(test_batch_to_visualize, learner, additional_caption="1: after pretraining")
        init_loss, fin_loss = learner.train_from_args(args)
        print(f"initial/final loss (joint, after pretraining):{init_loss:.3f}, {fin_loss:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, learner,
                                         additional_caption="2: after joint training")

    elif args.training_mode == "only_classification":
        utils.get_initial_expl_imgs(test_batch_to_visualize, learner, ImageHandler)
        init_loss_pretraining, final_loss_pretraining = learner.pretrain_from_args(args)
        print(f"initial/final loss (only classification): {init_loss_pretraining}, {final_loss_pretraining}")
        ImageHandler.add_gradient_images(test_batch=test_batch_to_visualize, learner=learner,
                                         additional_caption="after only-classification training")
        print(f"initial/final loss (pretraining):{init_loss_pretraining:.3f}, {final_loss_pretraining:3f}")
        ImageHandler.add_gradient_images(test_batch=test_batch_to_visualize, learner=learner, additional_caption="1: after pretraining", save=True)

    elif args.training_mode == "test":
        assert args.model_pt is not None
        print("Loading and evaluating specified model ...")
        learner.load_state(args.model_pt)
        test_acc = utils.compute_accuracy(learner.classifier, loaders.test)
        print(f"Test accuracy of {args.model_pt}: \n{100*test_acc}")

    elif args.training_mode == 'faithfulness':
        assert args.model_pt is not None
        print("Loading and evaluating specified model ...")
        learner.load_state(args.model_pt)
        utils_faithfulness.comp_faithfulness(learner, loaders, args)

    elif args.training_mode == "save_expls":
        assert args.model_pt is not None
        print("Loading and evaluating specified model ...")
        learner.load_state(args.model_pt)
        # generate all explanations of test set
        input_all, explanation_all, labels_all = learner.get_detached_labeled_explanation_input_batches(loaders.test)

        input_all = torch.cat(input_all).detach().cpu().numpy()
        explanation_all = torch.cat(explanation_all).detach().cpu().numpy()
        labels_all = torch.stack(labels_all).detach().cpu().numpy()

        # plot example explanations
        fig = utils.simple_plot(explanation_all)

        # save
        results_dict = {"inputs": input_all, "expls": explanation_all, "labels": labels_all}
        base_fp = os.path.join('runs',*args.model_pt.split('runs')[-1].split(os.path.sep)[:-1],'')
        with open(base_fp + 'test_expls.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
        fig.savefig(base_fp + 'test_expls.png', bbox_inches='tight')

    else:
        raise ValueError(f'Invalid training mode "{args.training_mode}"!')
    print(utils.colored(0, 200, 0, "Finished!"))


if __name__ == '__main__':
    run_experiments()
