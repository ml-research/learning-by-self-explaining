"""
Common functions for visualization in different ipython notebooks
"""
import os
import random
import numpy as np
import random
import torch

N_CLASSES = 200
N_ATTRIBUTES = 312

def sample_files(class_label, img_dir='CUB_200_2011/images/', number_of_files=10):
    """
    Given a class id, extract the path to the corresponding image folder and sample number_of_files randomly from that folder
    """
    class_to_folder = dict()
    for folder in os.listdir(img_dir):
        class_id = int(folder.split('.')[0])
        class_to_folder[class_id - 1] = os.path.join(img_dir, folder)

    folder = class_to_folder[class_label]
    class_files = random.sample(os.listdir(folder), number_of_files)
    class_files = [os.path.join(folder, f) for f in class_files]
    return class_files


def get_attribute_groupings():
    attr_group_dict = dict()
    curr_group_idx = 0
    with open('CUB_200_2011/attributes/attributes.txt', 'r') as f:
        all_lines = f.readlines()
        line0 = all_lines[0]
        prefix = line0.split()[1][:10]
        attr_group_dict[curr_group_idx] = [0]
        for i, line in enumerate(all_lines[1:]):
            curr = line.split()[1][:10]
            if curr != prefix:
                curr_group_idx += 1
                prefix = curr
                attr_group_dict[curr_group_idx] = [i + 1]
            else:
                attr_group_dict[curr_group_idx].append(i + 1)

    return attr_group_dict


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(16)
