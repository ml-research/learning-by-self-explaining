import torch
import numpy as np
import re

from CUB.dataset import PROP_STR_GROUPED, CATEGORY_STR, PROP_STR_GROUPED_FLAT

MAX_OBJS = 1
N_ATTR = len(PROP_STR_GROUPED_FLAT)#np.sum([len(PROP_STR_GROUPED[i]) for i in range(len(PROP_STR_GROUPED))])
# CATEGORY_PROP_ATTRID_DICT = generate_cat_prop_attrid_dict()
# a dictionary thaat has the category names as dictionary key and as values a tuple consisting of the proposition
# name of that category and the corresponding attribute id of the flat attribute vector
CATEGORY_PROP_ATTRID_DICT = {}
running_attrid = 0
for cat_id in range(len(CATEGORY_STR)):
	tmp_d = {}
	for i in range(len(PROP_STR_GROUPED[cat_id])):
		tmp_d[PROP_STR_GROUPED[cat_id][i]] = running_attrid
		running_attrid += 1
	CATEGORY_PROP_ATTRID_DICT[CATEGORY_STR[cat_id]] = tmp_d


def extract_object_numbers(rule):
	obj_in_str = 'in\(O\d,X\)'
	obj_strs = re.findall(obj_in_str, rule)

	list_objs = []
	for obj_str in obj_strs:
		nums = re.findall('\d', obj_str)
		assert len(nums) == 1
		list_objs.append(int(re.findall('\d', obj_str)[0]))

	return list_objs


def gen_mask(rule, max_objs, rrr):
	# get object ids that are present in rule
	list_objs = extract_object_numbers(rule)

	# split the rule into the individual predicates
	attr_list_unparse = rule.split('.')[0].split(':-')[-1].split('),')

	mask = np.zeros((max_objs, N_ATTR))
	# iterate over the objects and detect if that object is referenced in a predicate
	for obj_id in list_objs:

		for attr_unparse in attr_list_unparse:

			if f"O{obj_id}" in attr_unparse:
				# extract predicate name from string
				pred = attr_unparse.split(f"O{obj_id}")[0].split('(')[0]
				if pred in CATEGORY_STR:
					# extract attribute from the string
					attr = attr_unparse.split(f"O{obj_id}")[-1].split(',')[-1]
					if ')' in attr:
						attr = attr.split(')')[0]
					attr_idx = CATEGORY_PROP_ATTRID_DICT[pred][attr]
					# set mask at attributes location to True
					if rrr:
						mask[:, attr_idx] = 1 # mark whole attribute over all objects
					else:
						mask[obj_id - 1, attr_idx] = 1 # mark attribute only for particular object

	return mask


def gen_masks_from_list_of_rules(list_of_rules, max_objs=10, rrr=False):
	class_masks = []
	class_masks_inv = []
	# create a mask for each class
	for class_id in range(len(list_of_rules)):

		list_of_class_rules = list_of_rules[class_id]

		# if there are multiple rules for a class, concatenate them together
		class_mask = np.zeros(((max_objs, N_ATTR)))

		for rule in list_of_class_rules:
			mask = gen_mask(rule, max_objs, rrr)
			class_mask += mask

		# convert to boolean, as through the previous summation values may be larger than 1
		class_mask = class_mask.astype('bool')
		# invert the mask
		class_mask_inv = np.invert(class_mask)

		# convert back to float
		class_masks.append(class_mask.astype('float'))
		class_masks_inv.append(class_mask_inv.astype('float'))

	return torch.tensor(np.array(class_masks)), torch.tensor(np.array(class_masks_inv))