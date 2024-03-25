"""
General utils for training, evaluation and data loading
"""
import os

import PIL.ImageChops
import torch
import pickle
import numpy as np
from torchvision.io import read_image
from torchvision.io import ImageReadMode
import torchvision.transforms as T
import torch.nn as nn
from scipy.special import softmax

from PIL import Image
from CUB.config import BASE_DIR, N_ATTRIBUTES
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader
from CUB.data_processing import PAPER_MASK

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------------------------------------------------------------------------------------------------- #
# preprocessing for logic predicate names etc.
PROP_STR_GROUPED = [['curvedupordown', 'dagger', 'hooked', 'needle', 'hookedseabird', 'spatulate', 'allpurpose', 'cone', 'specialized'], ['blue', 'brown', 'iridescent', 'purple', 'rufous', 'grey', 'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white', 'red', 'buff'], ['blue', 'brown', 'iridescent', 'purple', 'rufous', 'grey', 'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white', 'red', 'buff'], ['blue', 'brown', 'iridescent', 'purple', 'rufous', 'grey', 'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white', 'red', 'buff'], ['solid', 'spotted', 'striped', 'multicolored'], ['blue', 'brown', 'iridescent', 'purple', 'rufous', 'grey', 'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white', 'red', 'buff'], ['forkedtail', 'roundedtail', 'notchedtail', 'fanshapedtail', 'pointedtail', 'squaredtail'], ['blue', 'brown', 'iridescent', 'purple', 'rufous', 'grey', 'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white', 'red', 'buff'], ['spotted', 'malar', 'crested', 'masked', 'uniquepattern', 'eyebrow', 'eyering', 'plain', 'eyeline', 'striped', 'capped'], ['blue', 'brown', 'iridescent', 'purple', 'rufous', 'grey', 'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white', 'red', 'buff'], ['blue', 'brown', 'iridescent', 'purple', 'rufous', 'grey', 'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white', 'red', 'buff'], ['blue', 'brown', 'purple', 'rufous', 'grey', 'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white', 'red', 'buff'], ['aboutthesameashead', 'longerthanhead', 'shorterthanhead'], ['blue', 'brown', 'iridescent', 'purple', 'rufous', 'grey', 'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white', 'red', 'buff'], ['blue', 'brown', 'iridescent', 'purple', 'rufous', 'grey', 'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white', 'red', 'buff'], ['blue', 'brown', 'iridescent', 'purple', 'rufous', 'grey', 'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white', 'red', 'buff'], ['blue', 'brown', 'iridescent', 'purple', 'rufous', 'grey', 'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white', 'red', 'buff'], ['roundedwings', 'pointedwings', 'broadwings', 'taperedwings', 'longwings'], ['large1632in', 'small59in', 'verylarge3272in', 'medium916in', 'verysmall35in'], ['uprightperchingwaterlike', 'chickenlikemarsh', 'longleggedlike', 'ducklike', 'owllike', 'gulllike', 'hummingbirdlike', 'pigeonlike', 'treeclinginglike', 'hawklike', 'sandpiperlike', 'uplandgroundlike', 'swallowlike', 'perchinglike'], ['solid', 'spotted', 'striped', 'multicolored'], ['solid', 'spotted', 'striped', 'multicolored'], ['solid', 'spotted', 'striped', 'multicolored'], ['blue', 'brown', 'iridescent', 'purple', 'rufous', 'grey', 'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white', 'red', 'buff'], ['blue', 'brown', 'iridescent', 'purple', 'rufous', 'grey', 'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white', 'red', 'buff'], ['blue', 'brown', 'iridescent', 'purple', 'rufous', 'grey', 'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white', 'red', 'buff'], ['blue', 'brown', 'iridescent', 'purple', 'rufous', 'grey', 'yellow', 'olive', 'green', 'pink', 'orange', 'black', 'white', 'red', 'buff'], ['solid', 'spotted', 'striped', 'multicolored']]
CATEGORY_STR = ['hasbillshape', 'haswingcolor', 'hasupperpartscolor', 'hasunderpartscolor', 'hasbreastpattern', 'hasbackcolor', 'hastailshape', 'hasuppertailcolor', 'hasheadpattern', 'hasbreastcolor', 'hasthroatcolor', 'haseyecolor', 'hasbilllength', 'hasforeheadcolor', 'hasundertailcolor', 'hasnapecolor', 'hasbellycolor', 'haswingshape', 'hassize', 'hasshape', 'hasbackpattern', 'hastailpattern', 'hasbellypattern', 'hasprimarycolor', 'haslegcolor', 'hasbillcolor', 'hascrowncolor', 'haswingpattern']

PROP_STR_GROUPED_MASK = []
CATEGORY_STR_MASK = CATEGORY_STR.copy()
running_prop_id = 0
for cat_id in range(len(CATEGORY_STR)):
    cat_list = []
    for prop_id in range(len(PROP_STR_GROUPED[cat_id])):
        if running_prop_id in PAPER_MASK:
            cat_list.append(PROP_STR_GROUPED[cat_id][prop_id])
        running_prop_id += 1
    if len(cat_list) == 0:
        CATEGORY_STR_MASK.remove(CATEGORY_STR[cat_id])
    else:
        PROP_STR_GROUPED_MASK.append(cat_list)

PROP_STR_GROUPED = PROP_STR_GROUPED_MASK
CATEGORY_STR = CATEGORY_STR_MASK

PROP_STR_GROUPED_FLAT = [item for sublist in PROP_STR_GROUPED for item in sublist]
CATEGORY_IDS = np.cumsum([len(PROP_STR_GROUPED[i]) for i in range(len(CATEGORY_STR))])
CATEGORY_IDS = np.insert(CATEGORY_IDS, 0, 0)
PROP_STR_BY_ID = [(i, cat_id) for cat_id in range(len(CATEGORY_STR)) for i in range(len(PROP_STR_GROUPED[cat_id]))]

# a dictionary that has the category names as dictionary key and as values a tuple consisting of the proposition
# name of that category and the corresponding attribute id of the flat attribute vector
CATEGORY_PROP_ATTRID_DICT = {}
running_attrid = 0
for cat_id in range(len(CATEGORY_STR)):
    tmp_d = {}
    for i in range(len(PROP_STR_GROUPED[cat_id])):
        tmp_d[PROP_STR_GROUPED[cat_id][i]] = running_attrid
        running_attrid += 1
    CATEGORY_PROP_ATTRID_DICT[CATEGORY_STR[cat_id]] = tmp_d

# -------------------------------------------------------------------------------------------------------------------- #


class CUBDataset(Dataset):
	"""
	Returns a compatible Torch Dataset object customized for the CUB dataset
	"""

	def __init__(self, pkl_file_paths, use_attr, no_img, image_dir, n_class, perc=1.0, transform=None):
		"""
		Arguments:
		pkl_file_paths: list of full path to all the pkl data
		use_attr: whether to load the attributes (e.g. False for simple finetune)
		no_img: whether to load the images (e.g. False for A -> Y model)
		image_dir: default = 'images'. Will be append to the parent dir
		transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
		"""
		self.data = []
		self.is_train = any(["train" in path for path in pkl_file_paths])
		if not self.is_train:
			assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])

		for file_path in pkl_file_paths:
			self.data.extend(pickle.load(open(file_path, 'rb')))

		self.transform = transform
		self.use_attr = use_attr
		self.no_img = no_img
		self.image_dir = image_dir
		self.perc = perc
		self.class_expl_feedback_masks = None
		self.class_expl_feedback_masks_inv = None
		self.use_softmax = False

		self.n_class = n_class
		if self.n_class is None:
			self.n_class = len(np.unique([self.data[i]['class_label'] for i in range(len(self.data))]))

		rel_inds = [i for i in range(len(self.data)) if self.data[i]['class_label'] < self.n_class]
		self.img_class_ids = [self.data[i]['class_label'] for i in rel_inds
							  if self.data[i]['class_label'] < self.n_class]

		# a list of all sample ids, unless otherwise specified this is just a list with ints from 0 to the number
		# of samples
		self.sample_ids = rel_inds
		self.sample_ids = self.sample_ids[:int(self.perc * len(self.sample_ids))]
		self.img_class_ids = self.img_class_ids[:int(self.perc * len(self.img_class_ids))]
		# print(f"{len(self.sample_ids)} samples")
		self.full_sample_ids = self.sample_ids.copy()

	def __len__(self):
		return len(self.sample_ids)

	def __getitem__(self, idx):
		id = self.sample_ids[idx]
		img_data = self.data[id]
		img_path = img_data['img_path']
		class_label = img_data['class_label']
		attr_label = img_data['attribute_label']

		if self.no_img:
			if self.class_expl_feedback_masks is not None:
				proposed_table_expl = self.class_expl_feedback_masks[class_label]
				proposed_table_expl_inv = self.class_expl_feedback_masks_inv[class_label]
				return attr_label, class_label, proposed_table_expl, proposed_table_expl_inv
			else:
				return attr_label, class_label
		else:

			img = read_image(img_path, ImageReadMode.RGB)
			img = img.to(device)
			img = img/255

			if self.transform:
				img = self.transform(img)

		if self.use_attr:
			return img, class_label, attr_label
		else:
			return img, class_label

	def update_pos_sample_ids(self, class_id):
		rel_ids = np.where(np.array(self.img_class_ids) == class_id)[0]
		self.sample_ids = list(np.array(self.full_sample_ids)[rel_ids])

	def update_neg_sample_ids(self, class_id):
		rel_ids = np.where(np.array(self.img_class_ids) != class_id)[0]
		self.sample_ids = list(np.array(self.full_sample_ids)[rel_ids])

	def reset_sample_ids(self):
		self.sample_ids = self.full_sample_ids


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Arguments:
		indices (list, optional): a list of indices
		num_samples (int, optional): number of samples to draw
	"""

	def __init__(self, dataset, indices=None):
		# if indices is not provided,
		# all elements in the dataset will be considered
		self.indices = list(range(len(dataset))) \
			if indices is None else indices

		# if num_samples is not provided,
		# draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices)

		# distribution of classes in the dataset
		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			if label in label_to_count:
				label_to_count[label] += 1
			else:
				label_to_count[label] = 1

		# weight for each sample
		weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
				   for idx in self.indices]
		self.weights = torch.DoubleTensor(weights)

	def _get_label(self, dataset, idx):  # Note: for single attribute dataset
		return dataset.data[idx]['attribute_label'][0]

	def __iter__(self):
		idx = (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))
		return idx

	def __len__(self):
		return self.num_samples


def load_data(pkl_paths, use_attr, no_img, batch_size, image_dir='images', resol=299, n_class=None, perc=1.0):
	"""
	Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
	Loads data with transformations applied, and upsample the minority class if there is class imbalance and weighted loss is not used
	NOTE: resampling is customized for first attribute only, so change sampler.py if necessary
	"""
	is_training = any(['train.pkl' in f for f in pkl_paths])
	if is_training:
		transform = nn.Sequential(
			T.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
			T.RandomResizedCrop(resol),
			T.RandomHorizontalFlip(),
			T.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
		)
	else:
		transform = nn.Sequential(
			T.CenterCrop(resol),
			T.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
		)

	dataset = CUBDataset(pkl_paths, use_attr, no_img, image_dir, n_class, perc=perc, transform=transform)
	if is_training:
		drop_last = False
		shuffle = True
	else:
		drop_last = False
		shuffle = False
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
	return loader


def get_loaders(base_dir, args):
	train_data_path = os.path.join(base_dir, args.data_dir, 'train.pkl')
	val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
	test_data_path = train_data_path.replace('train.pkl', 'test.pkl')

	train_loader = load_data([train_data_path], args.use_attr, args.no_img, args.batch_size,
	                         image_dir=args.image_dir, n_class=args.n_imgclasses, perc=args.perc)
	train_val_loader = load_data([train_data_path, val_data_path], args.use_attr, args.no_img, args.batch_size,
	                             image_dir=args.image_dir, n_class=args.n_imgclasses, perc=args.perc)
	val_loader = load_data([val_data_path], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
	                       n_class=args.n_imgclasses, perc=args.perc)
	test_loader = load_data([test_data_path], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
	                        n_class=args.n_imgclasses)

	loaders = (train_loader, train_val_loader, val_loader, test_loader)
	return loaders


def find_class_imbalance(pkl_file, multiple_attr=False, attr_idx=-1):
	"""
	Calculate class imbalance ratio for binary attribute labels stored in pkl_file
	If attr_idx >= 0, then only return ratio for the corresponding attribute id
	If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
	"""
	imbalance_ratio = []
	data = pickle.load(open(os.path.join(BASE_DIR, pkl_file), 'rb'))
	n = len(data)
	n_attr = len(data[0]['attribute_label'])
	if attr_idx >= 0:
		n_attr = 1
	if multiple_attr:
		n_ones = [0] * n_attr
		total = [n] * n_attr
	else:
		n_ones = [0]
		total = [n * n_attr]
	for d in data:
		labels = d['attribute_label']
		if multiple_attr:
			for i in range(n_attr):
				n_ones[i] += labels[i]
		else:
			if attr_idx >= 0:
				n_ones[0] += labels[attr_idx]
			else:
				n_ones[0] += sum(labels)
	for j in range(len(n_ones)):
		imbalance_ratio.append(total[j] / n_ones[j] - 1)
	if not multiple_attr:  # e.g. [9.0] --> [9.0] * 312
		imbalance_ratio *= n_attr
	return imbalance_ratio
