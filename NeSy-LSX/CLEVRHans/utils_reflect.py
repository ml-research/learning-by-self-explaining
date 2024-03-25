import torch
import numpy as np
import itertools
import os
import shutil
from tqdm import tqdm

import utils as utils


PROP_STR_GROUPED = [['sphere', 'cube', 'cylinder'],
                    ['large', 'small'],
                    ['rubber', 'metal'],
                    ['cyan', 'blue', 'yellow', 'purple', 'red', 'green', 'gray', 'brown']]
CATEGORY_STR = ['shape', 'size', 'material', 'color']
PROP_STR = [prop for list in PROP_STR_GROUPED for prop in list]


def propositionalise(nn_model, loader, iter, args):

    # get and store symbolic explanations of concept learner
    attr_saliencies, gt_classes, pred_classes, pred_attrs = generate_explanations(nn_model, loader, args)

    # generate all hypothesized rules from the attributes saliency maps
    test_hypothesis, test_hypothesis_ints, gt_class_id_per_rule = gen_propositions(
        attr_saliencies, gt_classes, pred_classes, pred_attrs, args.prop_thresh
    )

    # remove all duplicates
    test_hypothesis, test_hypothesis_ints, gt_class_id_per_rule = remove_duplicates(
        test_hypothesis, test_hypothesis_ints, gt_class_id_per_rule, args.n_imgclasses
    )

    test_hypothesis = rename_clause_heads(test_hypothesis, gt_class_id_per_rule, args.n_imgclasses)

    # write the clauses and predicates to txt file
    save_to_txt_by_iter(test_hypothesis, iter, args)
    save_to_txt(test_hypothesis, np.array(gt_class_id_per_rule), args)
    print("Proposal clauses generated and stored ...")


def save_to_txt_by_iter(test_hypothesis, iter, args):
    save_dir = os.path.join(args.log_dir, "proposed_clauses")
    os.makedirs(save_dir, exist_ok=True)

    textfile = open(os.path.join(save_dir, f"clauses_{iter}.txt"), "w")
    for rule in test_hypothesis:
        textfile.write(rule + "\n")
    textfile.close()


def save_to_txt(test_hypothesis, gt_class_id_per_rule, args):
    save_dir = os.path.join(args.log_dir, "proposed_clauses")

    for class_id in range(args.n_imgclasses):

        rel_ids = np.where(gt_class_id_per_rule == class_id)[0]

        base_pth = os.path.join(save_dir, f"{args.dataset_type}-{class_id}")
        os.makedirs(base_pth, exist_ok=True)

        textfile = open(os.path.join(base_pth, f"clauses.txt"), "w")
        for i in rel_ids:
            rule = test_hypothesis[i]
            textfile.write(rule + "\n")
        textfile.close()

        textfile = open(os.path.join(base_pth, f"preds.txt"), "w")
        # for rule in test_hypothesis:
        # pred_str = rule.split('(X)')[0] + ":1:image"
        pred_str = "pos:1:image"
        textfile.write(pred_str + "\n")
        textfile.close()

        # copy base lang into each class folder
        base_nsfr_lang_path = os.path.join('NSFRAlpha', 'data', 'lang', args.dataset_type)
        shutil.copyfile(os.path.join(base_nsfr_lang_path, 'neural_preds.txt'),
                        os.path.join(base_pth, 'neural_preds.txt'))
        shutil.copyfile(os.path.join(base_nsfr_lang_path, 'consts.txt'),
                        os.path.join(base_pth, 'consts.txt'))

def generate_explanations(nn_model, loader, args):

    nn_model.eval()

    # B = loader.batch_size
    # loader.batch_size = 1

    # create empty tensors for storing the data
    attr_saliencies = torch.empty((loader.dataset.__len__(), args.n_slots, args.n_attr+1))
    gt_classes = torch.empty(loader.dataset.__len__())
    pred_classes = torch.empty(loader.dataset.__len__())
    pred_attrs = torch.empty((loader.dataset.__len__(), args.n_slots, args.n_attr))
    for i, sample in enumerate(loader):
        # if i > 0:
        #     break

        # input is either a set or an image
        imgs, gt_attr, gt_class, _, _, _, _, _ = map(lambda x: x.to(args.device), sample)
        # img_class_ids = img_class_ids.long()

        # forward evaluation through the network
        output_cls, output_attr, obj_pres = nn_model.forward(imgs)
        _, preds = torch.max(output_cls, 1)

        #     # convert sorting gt target set and gt table explanations to match the order of the predicted table
        #     attrs, match_ids = utils.hungarian_matching(output_attr.to('cpu'), attrs)
        # #     # table_expls = table_expls[:, match_ids][range(table_expls.shape[0]), range(table_expls.shape[0])]

        # get explanations of set classifier
        symb_expl = utils.generate_intgrad_captum_table(nn_model.set_cls, gt_attr, obj_pres,
                                                               gt_class, device=args.device)

        # stack over batches
        attr_saliencies[(i*loader.batch_size):(i*loader.batch_size + loader.batch_size)] = symb_expl.detach()
        gt_classes[(i*loader.batch_size):(i*loader.batch_size + loader.batch_size)] = gt_class.detach()
        pred_classes[(i*loader.batch_size):(i*loader.batch_size + loader.batch_size)] = preds.detach()
        pred_attrs[(i*loader.batch_size):(i*loader.batch_size + loader.batch_size)] = output_attr.detach()

    return attr_saliencies, gt_classes, pred_classes, pred_attrs


def gen_propositions(attr_saliencies, gt_classes, pred_classes, pred_attrs, threshold):

    test_hypothesis_all = []
    test_hypothesis_ints_all = []
    class_id_per_rule_all = []
    for sample_id in tqdm(range(len(gt_classes))):
        pred_class_id = pred_classes[sample_id].detach().cpu().numpy()
        gt_class_id = gt_classes[sample_id].detach().cpu().numpy()
        if pred_class_id == gt_class_id:
            bin_pos_sals, bin_neg_sals = get_pos_and_neg_sal(attrs=pred_attrs, attr_saliencies=attr_saliencies,
                                                             sample_id=sample_id, threshold=threshold)
            unique_test_hypothesis, sorted_test_hypothesis_ints = get_unique_hypothesised_rules(bin_pos_sals,
                                                                                                bin_neg_sals,
                                                                                                class_id=gt_class_id)
            test_hypothesis_all.extend(unique_test_hypothesis)
            test_hypothesis_ints_all.extend(sorted_test_hypothesis_ints)
            class_id_per_rule_all.extend(gt_class_id * np.ones(len(unique_test_hypothesis), dtype=int))

    print(f"{len(test_hypothesis_all)} initial hypothesised rules")

    return test_hypothesis_all, test_hypothesis_ints_all, class_id_per_rule_all


def remove_duplicates(test_hypothesis_all, test_hypothesis_ints_all, class_id_per_rule_all, n_classes):
    # sort the sublists for permutation invariant comparisons
    sorted_test_hypothesis_ints_all = [sort_list_of_lists(l) for l in test_hypothesis_ints_all]

    # check for cooccurrences and store the indices of the first occurrence of each entry per class
    print("Removing duplicates ...")
    keep_ids = []
    for class_id in tqdm(range(n_classes)):
        tmp_ids = np.where(np.array(class_id_per_rule_all) == class_id)[0]
        tmp_class_list = [sorted_test_hypothesis_ints_all[i] for i in tmp_ids]
        seen = []
        for i, l in enumerate(tmp_class_list):
            if seen.count(l) == 0:
                keep_ids.append(tmp_ids[i])
                seen.append(l)

    # apply the entry ids to be kept to the propositional form list to get unique propositional clauses
    unique_test_hypothesis_all = [test_hypothesis_all[i] for i in keep_ids]
    unique_sorted_test_hypothesis_ints_all = [sorted_test_hypothesis_ints_all[i] for i in keep_ids]
    unique_class_id_per_rule_all = [class_id_per_rule_all[i] for i in keep_ids]

    print(f"{len(unique_test_hypothesis_all)} secondary hypothesised rules (without duplicates)")

    # now remove permuted, but same clauses
    # print("Removing duplicates with permuted attirbutes ...")
    # remove_ids = get_ids_of_permutated_clauses(unique_test_hypothesis_all, unique_class_id_per_rule_all, n_classes)
    # for remove_id in tqdm(remove_ids):
    #     unique_test_hypothesis_all.pop(remove_id)
    #     unique_sorted_test_hypothesis_ints_all.pop(remove_id)
    #     unique_class_id_per_rule_all.pop(remove_id)
    #
    # print(f"{len(unique_test_hypothesis_all)} tertiary hypothesised rules (without object permuted duplicates)")

    return unique_test_hypothesis_all, unique_sorted_test_hypothesis_ints_all, unique_class_id_per_rule_all


def rename_clause_heads(unique_test_hypothesis_all, unique_class_id_per_rule_all, n_classes):
    # rename the clause head by indexing
    for class_id in range(n_classes):
        ids = np.where(np.array(unique_class_id_per_rule_all) == class_id)[0]
        for i, idx in enumerate(ids):
            unique_test_hypothesis_all[idx] = unique_test_hypothesis_all[idx].replace(
                f"ch{class_id}(X)",
                f"pos(X)"
            )
    return unique_test_hypothesis_all


def get_ids_of_permutated_clauses(unique_test_hypothesis_all, unique_class_id_per_rule_all, n_classes):
    # iterate over classes and remove clauses that are the same just with different object permutations
    remove_clause_ids = []
    for class_id in range(n_classes):
        class_ids = np.where(np.array(unique_class_id_per_rule_all) == class_id)[0]

        #     class_unique_test_hypothesis_all = unique_test_hypothesis_all[ids]

        class_unique_test_hypothesis_all_as_atoms = []
        class_n_objs_per_unique_test_hypothesis = []

        for clause_id in class_ids:
            unique_test_hypothesis = unique_test_hypothesis_all[clause_id]
            # extract each individual atom from the clause string
            ind_atoms = unique_test_hypothesis.split(':-')[-1].split(').')[0].split('),')
            ind_atoms = [atom + ')' for atom in ind_atoms]
            class_unique_test_hypothesis_all_as_atoms.append(ind_atoms)

            # get number of objects in each clause
            class_n_objs_per_unique_test_hypothesis.append(
                len(np.unique([int(i) for i in unique_test_hypothesis.split(':-')[-1] if i.isdigit()]))
            )

        class_remove_clause_ids = []
        for clause_id in range(len(class_unique_test_hypothesis_all_as_atoms)):
            n_objs = class_n_objs_per_unique_test_hypothesis[clause_id]
            if n_objs > 1:
                # create a mapping of how to replace the obj ids
                replace_lists = []
                list_objs = list(1 + np.arange(0, n_objs))
                permutations = list(itertools.permutations(list_objs))
                for permutation in permutations:
                    replace_lists.append(np.array([list_objs, permutation]).T)

                # go through all atoms and replace the obj ids accordingly
                permuted_clauses = []
                for replace_id in range(len(replace_lists)):
                    tmp_clause = []
                    for atom in class_unique_test_hypothesis_all_as_atoms[clause_id]:
                        tmp_atom = ''
                        for s in atom:
                            if s.isdigit() and int(s) in list_objs:
                                s = str(replace_lists[replace_id][int(s) - 1][1])
                            tmp_atom += s
                        tmp_clause.append(tmp_atom)
                    permuted_clauses.append(tmp_clause)

                # iterate over permuted clauses, if more than one of the permuted clauses occurs in the list of
                # hypothesized clauses, then there is another permutation of the current clause in the list of
                # hypothesized clauses
                cooccurance_ids = False * np.zeros(len(class_unique_test_hypothesis_all_as_atoms))
                for permuted_clause in permuted_clauses:
                    cooccurance_ids += np.array(
                        [
                            all(elem in permuted_clause for elem in class_unique_test_hypothesis_all_as_atoms[i])
                            and len(class_unique_test_hypothesis_all_as_atoms[i]) == len(permuted_clause)
                            for i in range(len(class_unique_test_hypothesis_all_as_atoms))
                        ]
                    )

                # if more than one permuted clause appears: always keep only the first occurance of these
                n_permutations_occur = len(np.where(cooccurance_ids)[0])
                if n_permutations_occur > 1:
                    print(clause_id)
                    class_remove_clause_ids.append(np.where(cooccurance_ids)[0][1:])

        # remove duplicates
        class_remove_clause_ids = np.unique(class_remove_clause_ids)
        if len(class_remove_clause_ids) > 0:
            remove_clause_ids.append(class_ids[class_remove_clause_ids][0])

    return remove_clause_ids


def sort_list_of_lists(l):
    return sorted([sorted(sub_l) for sub_l in l])


def get_category_string(string):
    for i, cat_list in enumerate(PROP_STR_GROUPED):
        if string in cat_list:
            return i
    else:
        raise RuntimeError("property string is not in PROP_STR_GROUPED!")
        return False


def get_obj_string(obj_id, rel_attr_ids, max_attrs):
    # def get_obj_string(obj_id: int, rel_attr_ids: list[int], max_attrs: int):
    # collect all attributes of this object up to max_attrs
    # get set of possible combinations of attribute ids given max_attrs
    comb_ids = list(itertools.combinations(rel_attr_ids, max_attrs))
    obj_strs = []
    attr_ids = []
    for i, perm in enumerate(comb_ids):
        obj_str = ""
        tmp_arr = []
        for j in perm:
            tmp_arr.append(j)
            obj_str += (
                f",{CATEGORY_STR[get_category_string(PROP_STR[j])]}"
                f"(O{obj_id},{PROP_STR[j]})"
            )
        attr_ids.append(tmp_arr)
        obj_strs.append(obj_str)
    return obj_strs, attr_ids


def unique_combinations(elements, k) -> list:
    # def unique_combinations(elements: list[int], k: int) -> list[tuple[int, int]]:
    """
    Precondition: `elements` does not contain duplicates.
    Postcondition: Returns unique combinations of length k from `elements`.
    """
    return list(itertools.combinations(elements, k))


def get_pos_and_neg_sal(attrs: torch.Tensor, attr_saliencies: torch.Tensor,
                        # pred_class_id: int, class_id: int,
                        sample_id: int, threshold: float,
                        # verbose: int
                        ):
    obj_pres = attr_saliencies[:, :, 0]
    # remove position variables (idx 1-3) and object presence (idx 0)
    # plot_attrs = attrs[:, :, 3:]
    plot_attr_saliencies = attr_saliencies[:, :, 4:]

    # get only positive explanations
    pos_saliencies = torch.clone(plot_attr_saliencies)
    pos_saliencies[pos_saliencies < 0] = 0.
    # but remove values too close to 0
    pos_saliencies[pos_saliencies < threshold] = 0.

    # get only negative explanations
    # plot_attr_saliencies
    neg_saliencies = torch.clone(plot_attr_saliencies)
    neg_saliencies[neg_saliencies > 0] = 0.
    # but remove values too close to 0
    neg_saliencies[neg_saliencies > -threshold] = 0.

    # binarize thresholded negative and positive attributions
    bin_pos_sals = torch.clone(pos_saliencies)[sample_id]
    bin_pos_sals[bin_pos_sals > 0] = 1
    bin_pos_sals = bin_pos_sals.detach().cpu().numpy()

    bin_neg_sals = torch.clone(neg_saliencies)[sample_id]
    bin_neg_sals[bin_neg_sals < 0] = 1
    bin_neg_sals = bin_neg_sals.detach().cpu().numpy()

    # if verbose:
    #     fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    #     ax = ax.flatten()
    #
    #     img0 = ax[0].imshow(plot_attrs[sample_id].detach().cpu().numpy(), cmap='binary')
    #     fig.colorbar(img0, ax=ax[0])
    #     ax[0].set_title((
    #         f"GT Features; Class: {class_id}; "
    #         f"Pred: {pred_class_id}")
    #     )
    #     ax[0].set_ylabel
    #     ax[0].set_xticks([])
    #
    #     #         fig.delaxes(ax[1])
    #     # unnormalize images
    #     img = imgs[sample_id]
    #     img = img / 2. + 0.5  # Rescale to [0, 1].
    #     img = np.array(transforms.ToPILImage()(img.cpu()).convert("RGB"))
    #     ax[1].imshow(img)
    #     ax[1].axis('off')
    #
    #     img2 = ax[2].imshow(pos_saliencies[sample_id].detach().cpu().numpy(), cmap='gist_yarg')
    #     fig.colorbar(img2, ax=ax[2])
    #     ax[2].set_title("Positive Attribution")
    #     ax[2].set_xticks([])
    #
    #     ax[3].imshow(bin_pos_sals, cmap='gist_yarg')
    #     ax[3].set_title("Binary Positive Attribution")
    #     ax[3].set_xticks([])
    #
    #     img4 = ax[4].imshow(neg_saliencies[sample_id].detach().cpu().numpy(), cmap='gray')
    #     fig.colorbar(img4, ax=ax[4])
    #     ax[4].set_title("Negative Attribution")
    #     ax[4].set_xticks(range(len(XTICKLABELS)))
    #     ax[4].set_xticklabels(XTICKLABELS, rotation=90)
    #
    #     ax[5].imshow(bin_neg_sals, cmap='gist_yarg')
    #     ax[5].set_title("Binary Negative Attribution")
    #     ax[5].set_xticks(range(len(XTICKLABELS)))
    #     ax[5].set_xticklabels(XTICKLABELS, rotation=90)
    #
    #     plt.show()

    return bin_pos_sals, bin_neg_sals


def get_unique_hypothesised_rules(bin_pos_sals, bin_neg_sals, class_id):
    """
    Receives the set of positive and negative saliency maps and returns a set of possible logical rules that can be
    found within the saliencies. Where a logical rule so far is only created as a conjunction of attribute values and
    up to three objects per rule.

    Returns set of logical rules created from a variety of combinations of the positive attribute saliencies.

    unique_test_hypothesis: contains all generated logical rules as strings
    sorted_test_hypothesis_ints: contains all generated logical rules as lists of lists of integers. E.g. [[2], [0, 7]]
    reads as object one has 2nd attribute and object two has 0 and 7th attribute.
    """
    test_hypothesis = []
    test_hypothesis_ints = []  # simple representation of attribute ids as ints

    # get ids of important objects and features
    pos_obj_ids, pos_attr_ids = np.where(bin_pos_sals)
    # ids of objects with at least one important attribute
    pos_single_obj_ids = np.unique(pos_obj_ids)
    # ids of attributes that are important
    pos_single_attr_ids = np.unique(pos_attr_ids)

    # neg_obj_ids, neg_attr_ids = np.where(bin_neg_sals)
    # neg_single_obj_ids = np.unique(neg_obj_ids)
    # neg_single_attr_ids = np.unique(neg_attr_ids)

    # get single object attributes
    for j in pos_single_obj_ids:
        tmp_ids = np.where(bin_pos_sals[j])[0]

        # allow for 1 to 4 attributes per object
        for max_attrs in range(1, 4):
            if len(tmp_ids) >= max_attrs:
                obj_strs, attr_ids = get_obj_string(obj_id=1, rel_attr_ids=tmp_ids, max_attrs=max_attrs)
                for (obj_str, attr_id) in zip(obj_strs, attr_ids):
                    tmp_str = f"ch{int(class_id)}(X):-in(O1,X){obj_str}."
                    test_hypothesis.append(tmp_str)
                    test_hypothesis_ints.append([attr_id])

    if len(pos_single_obj_ids) >= 2:
        pair_obj_ids = unique_combinations(pos_single_obj_ids, k=2)
        for pair_ids in pair_obj_ids:

            tmp_ids_1 = np.where(bin_pos_sals[pair_ids[0]])[0]
            tmp_ids_2 = np.where(bin_pos_sals[pair_ids[1]])[0]
            # allow for 1 to 4 attributes per object
            for max_attrs_1 in range(1, 4):
                for max_attrs_2 in range(1, 4):
                    if len(tmp_ids_1) >= max_attrs_1 and len(tmp_ids_2) >= max_attrs_2:
                        obj_strs_1, attr_ids_1 = get_obj_string(
                            obj_id=1, rel_attr_ids=tmp_ids_1, max_attrs=max_attrs_1
                        )
                        obj_strs_2, attr_ids_2 = get_obj_string(
                            obj_id=2, rel_attr_ids=tmp_ids_2, max_attrs=max_attrs_2
                        )
                        for (obj_str_1, attr_id_1) in zip(obj_strs_1, attr_ids_1):
                            for (obj_str_2, attr_id_2) in zip(obj_strs_2, attr_ids_2):
                                tmp_str = (
                                    f"ch{int(class_id)}(X):-"
                                    f"in(O1,X),in(O2,X){obj_str_1}{obj_str_2}."
                                )
                                test_hypothesis.append(tmp_str)
                                test_hypothesis_ints.append([attr_id_1, attr_id_2])

        if len(pos_single_obj_ids) >= 3:
            triple_obj_ids = unique_combinations(pos_single_obj_ids, k=3)
            for triple_ids in triple_obj_ids:

                tmp_ids_1 = np.where(bin_pos_sals[triple_ids[0]])[0]
                tmp_ids_2 = np.where(bin_pos_sals[triple_ids[1]])[0]
                tmp_ids_3 = np.where(bin_pos_sals[triple_ids[2]])[0]

                # allow for 1 to 4 attributes per object
                for max_attrs_1 in range(1, 4):
                    for max_attrs_2 in range(1, 4):
                        for max_attrs_3 in range(1, 4):
                            if len(tmp_ids_1) >= max_attrs_1 and len(tmp_ids_2) >= max_attrs_2 and len(tmp_ids_3) >= max_attrs_3:
                                obj_strs_1, attr_ids_1 = get_obj_string(
                                    obj_id=1, rel_attr_ids=tmp_ids_1, max_attrs=max_attrs_1
                                )
                                obj_strs_2, attr_ids_2 = get_obj_string(
                                    obj_id=2, rel_attr_ids=tmp_ids_2, max_attrs=max_attrs_2
                                )
                                obj_strs_3, attr_ids_3 = get_obj_string(
                                    obj_id=3, rel_attr_ids=tmp_ids_3, max_attrs=max_attrs_3
                                )
                                for (obj_str_1, attr_id_1) in zip(obj_strs_1, attr_ids_1):
                                    for (obj_str_2, attr_id_2) in zip(obj_strs_2, attr_ids_2):
                                        for (obj_str_3, attr_id_3) in zip(obj_strs_3, attr_ids_3):
                                            tmp_str = (
                                                f"ch{int(class_id)}(X):-"
                                                f"in(O1,X),in(O2,X),in(O3,X){obj_str_1}{obj_str_2}{obj_str_3}."
                                            )
                                            test_hypothesis.append(tmp_str)
                                            test_hypothesis_ints.append([attr_id_1, attr_id_2, attr_id_3])

    # for i, rule in enumerate(test_hypothesis):
    #     print(f"{rule}")

    # sort the sublists for permutation invariant comparisons
    sorted_test_hypothesis_ints = [sort_list_of_lists(l) for l in test_hypothesis_ints]

    # check for cooccurrences and store the indices of the first occurrence of each entry
    keep_ids = []
    seen = []
    for i, l in enumerate(sorted_test_hypothesis_ints):
        if seen.count(l) == 0:
            keep_ids.append(i)
            seen.append(l)

    # apply the entry ids to be kept to the propositional form list to get unique propositional clauses
    unique_test_hypothesis = [test_hypothesis[i] for i in keep_ids]
    sorted_test_hypothesis_ints = [sorted_test_hypothesis_ints[i] for i in keep_ids]

    #     # rename the clause head by indexing
    #     for i, rule in enumerate(unique_test_hypothesis):
    #         unique_test_hypothesis[i] = rule.replace(f"ch{img_class_ids[ID].cpu().numpy()}(X)",
    #                                                  f"ch{img_class_ids[ID].cpu().numpy()}_{i}(X)")

    return unique_test_hypothesis, sorted_test_hypothesis_ints
