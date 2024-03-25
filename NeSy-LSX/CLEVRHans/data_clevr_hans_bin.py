import os
import json

import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
import numpy as np

from pycocotools import mask as coco_mask

import data_clevr_hans as data

os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"

def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )


CLASSES = {
    "shape": ["sphere", "cube", "cylinder"],
    "size": ["large", "small"],
    "material": ["rubber", "metal"],
    "color": ["cyan", "blue", "yellow", "purple", "red", "green", "gray", "brown"],
}


class CLEVR_HANS_EXPL_bin_positive(data.CLEVR_HANS_EXPL):
    def __init__(self, base_path, split, perc=1., lexi=False, conf_vers='conf_2', pos_classid=0):
        self.pos_classid = pos_classid
        super(CLEVR_HANS_EXPL_bin_positive, self).__init__(base_path, split, perc, lexi, conf_vers)

    def prepare_scenes(self, scenes_json):
        img_ids = []
        scenes = []
        gt_img_expls = []
        gt_classes = []
        gt_symb_expls = []
        fnames = []
        for scene in scenes_json:
            if scene['class_id'] == self.pos_classid:
                fnames.append(os.path.join(self.images_folder, scene['image_filename']))
                # get global class id
                gt_classes.append(scene['class_id'])
                img_idx = scene["image_index"]

                objects = [self.object_to_fv(obj, scene['directions']) for obj in scene["objects"]]
                objects = torch.FloatTensor(objects).transpose(0, 1)

                # get gt image explanation based on the classification rule of the class label
                gt_img_expl_mask = self.get_img_expl_mask(scene)
                gt_img_expls.append(gt_img_expl_mask)

                num_objects = objects.size(1)
                # pad with 0s
                if num_objects < self.max_objects:
                    objects = torch.cat(
                        [
                            objects,
                            torch.zeros(objects.size(0), self.max_objects - num_objects),
                        ],
                        dim=1,
                    )

                # get gt table explanation based on the classification rule of the class label
                gt_table_expl_mask = self.get_table_expl_mask(objects, scene['class_id'])
                gt_symb_expls.append(gt_table_expl_mask)

                # fill in masks
                mask = torch.zeros(self.max_objects)
                mask[:num_objects] = 1

                # concatenate obj indication to end of object list
                objects = torch.cat((mask.unsqueeze(dim=0), objects), dim=0)

                img_ids.append(img_idx)
                scenes.append(objects.T)
        return img_ids, gt_classes, scenes, fnames, gt_img_expls, gt_symb_expls



class CLEVR_HANS_EXPL_bin_negative(data.CLEVR_HANS_EXPL):
    def __init__(self, base_path, split, perc=1., lexi=False, conf_vers='conf_2', pos_classid=0):
        self.pos_classid = pos_classid
        super(CLEVR_HANS_EXPL_bin_negative, self).__init__(base_path, split, perc, lexi, conf_vers)

    def prepare_scenes(self, scenes_json):
        img_ids = []
        scenes = []
        gt_img_expls = []
        gt_classes = []
        gt_symb_expls = []
        fnames = []
        for scene in scenes_json:
            if scene['class_id'] != self.pos_classid:
                fnames.append(os.path.join(self.images_folder, scene['image_filename']))
                # get global class id
                gt_classes.append(scene['class_id'])
                img_idx = scene["image_index"]

                objects = [self.object_to_fv(obj, scene['directions']) for obj in scene["objects"]]
                objects = torch.FloatTensor(objects).transpose(0, 1)

                # get gt image explanation based on the classification rule of the class label
                gt_img_expl_mask = self.get_img_expl_mask(scene)
                gt_img_expls.append(gt_img_expl_mask)

                num_objects = objects.size(1)
                # pad with 0s
                if num_objects < self.max_objects:
                    objects = torch.cat(
                        [
                            objects,
                            torch.zeros(objects.size(0), self.max_objects - num_objects),
                        ],
                        dim=1,
                    )

                # get gt table explanation based on the classification rule of the class label
                gt_table_expl_mask = self.get_table_expl_mask(objects, scene['class_id'])
                gt_symb_expls.append(gt_table_expl_mask)

                # fill in masks
                mask = torch.zeros(self.max_objects)
                mask[:num_objects] = 1

                # concatenate obj indication to end of object list
                objects = torch.cat((mask.unsqueeze(dim=0), objects), dim=0)

                img_ids.append(img_idx)
                scenes.append(objects.T)
        return img_ids, gt_classes, scenes, fnames, gt_img_expls, gt_symb_expls

