"""
Create variants of the initial CUB dataset
"""
import argparse
import copy
import os
import pickle
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.io import ImageReadMode
from torchvision.io import read_image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from CUB.config import N_ATTRIBUTES, N_CLASSES

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# not needed
def get_class_attributes_data(min_class_count, out_dir, modify_data_dir='', keep_instance_data=False):
    """
    Use train.pkl to aggregate attributes on class level and only keep those that are predominantly 1 for at least min_class_count classes
    Transform data in modify_data_dir file using the class attribute statistics and save the new dataset to out_dir
    If keep_instance_data is True, then retain the original values of the selected attributes. Otherwise, save aggregated class level attributes
    In our paper, we set min_class_count to be 10 and only use the following 112 attributes of indices 
    [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]
    """
    data = pickle.load(open('train.pkl', 'rb'))
    class_attr_count = np.zeros((N_CLASSES, N_ATTRIBUTES, 2))
    for d in data:
        class_label = d['class_label']
        certainties = d['attribute_certainty']
        for attr_idx, a in enumerate(d['attribute_label']):
            if a == 0 and certainties[attr_idx] == 1: #not visible
                continue
            class_attr_count[class_label][attr_idx][a] += 1

    class_attr_min_label = np.argmin(class_attr_count, axis=2)
    class_attr_max_label = np.argmax(class_attr_count, axis=2)
    equal_count = np.where(class_attr_min_label == class_attr_max_label) #check where 0 count = 1 count, set the corresponding class attribute label to be 1
    class_attr_max_label[equal_count] = 1

    attr_class_count = np.sum(class_attr_max_label, axis=0)
    mask = np.where(attr_class_count >= min_class_count)[0] #select attributes that are present (on a class level) in at least [min_class_count] classes
    class_attr_label_masked = class_attr_max_label[:, mask]
    if keep_instance_data:
        collapse_fn = lambda d: list(np.array(d['attribute_label'])[mask])
    else:
        collapse_fn = lambda d: list(class_attr_label_masked[d['class_label'], :])
    create_new_dataset(out_dir, 'attribute_label', collapse_fn, data_dir=modify_data_dir)


def change_img_dir_data(new_image_folder, datasets, data_dir='', out_dir='masked_datasets/'):
    """
    Change the prefix of img_path data in data_dir to new_image_folder
    """
    compute_fn = lambda d: os.path.join(new_image_folder, d['img_path'].split('/')[-2], d['img_path'].split('/')[-1]) 
    create_new_dataset(out_dir, 'img_path', datasets=datasets, compute_fn=compute_fn, data_dir=data_dir)

def create_logits_data(model_path, out_dir, data_dir='', use_sigmoid=False):
    """
    Replace attribute labels in data_dir with the logits output by the model from model_path and save the new data to out_dir
    """
    model = torch.load(model_path)
    model.to(device)
    get_logits_train = lambda d: inference(d['img_path'], model, use_sigmoid, is_train=True)
    get_logits_test = lambda d: inference(d['img_path'], model, use_sigmoid, is_train=False)
    create_new_dataset(out_dir, 'attribute_label', get_logits_train, datasets=['train'], data_dir=data_dir)
    create_new_dataset(out_dir, 'attribute_label', get_logits_test, datasets=['val', 'test'], data_dir=data_dir)

def inference(img_path, model, use_sigmoid, is_train, resol=299, layer_idx=None):
    """
    For a single image stored in img_path, run inference using model and return A\hat (if layer_idx is None) or values extracted from layer layer_idx 
    """
    model.eval()
    if is_train:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(resol),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])

    # # Trim unnecessary paths
    # try:
    #     idx = img_path.split('/').index('CUB_200_2011')
    #     img_path = '/'.join(img_path.split('/')[idx:])
    # except:
    #     img_path_split = img_path.split('/')
    #     split = 'train' if is_train else 'test'
    #     img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])

    img = read_image(img_path, ImageReadMode.RGB)
    img = img.to(device)
    img = img / 255
    img = transform(img).unsqueeze(0)
    if layer_idx is not None:
        all_mods = list(model.modules())
        cropped_model = torch.nn.Sequential(*list(model.children())[:layer_idx])  # nn.ModuleList(all_mods[:layer_idx])
        return cropped_model(img)

    outputs = model(img)
    if use_sigmoid:
        attr_outputs = [torch.nn.Sigmoid()(o) for o in outputs]
    else:
        attr_outputs = outputs

    attr_outputs = torch.cat([o.unsqueeze(1) for o in attr_outputs], dim=1).squeeze()
    return list(attr_outputs.data.cpu().numpy())

def create_new_dataset(out_dir, field_change, compute_fn, datasets=['train', 'val', 'test'], data_dir=''):
    """
    Generic function that given datasets stored in data_dir, modify/ add one field of the metadata in each dataset based on compute_fn
                          and save the new datasets to out_dir
    compute_fn should take in a metadata object (that includes 'img_path', 'class_label', 'attribute_label', etc.)
                          and return the updated value for field_change
    """

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for dataset in datasets:
        path = os.path.join(data_dir, dataset + '.pkl')
        if not os.path.exists(path):
            continue
        data = pickle.load(open(path, 'rb'))
        new_data = []
        for d in data:
            new_d = copy.deepcopy(d)
            new_value = compute_fn(d)
            if field_change in d:
                old_value = d[field_change]
                assert (type(old_value) == type(new_value))
            new_d[field_change] = new_value
            new_data.append(new_d)
        f = open(os.path.join(out_dir, dataset + '.pkl'), 'wb')
        pickle.dump(new_data, f)
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str,
                        choices=['ExtractConcepts', 'ChangeAdversarialDataDir'],
                        help='Name of experiment to run.')
    parser.add_argument('--model_path', type=str, help='Path of model')
    parser.add_argument('--out_dir', type=str, help='Output directory')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--adv_data_dir', type=str, help='Adversarial data directory')
    parser.add_argument('--train_splits', type=str, nargs='+', help='Train splits to use')
    parser.add_argument('--use_sigmoid', action='store_true', help='Use Sigmoid')
    args = parser.parse_args()

    if args.exp == 'ExtractConcepts':
        create_logits_data(args.model_path, args.out_dir, args.data_dir, args.use_sigmoid)
    elif args.exp == 'ChangeAdversarialDataDir':
        change_img_dir_data(args.adv_data_dir, datasets=args.train_splits, data_dir=args.data_dir, out_dir=args.out_dir)
