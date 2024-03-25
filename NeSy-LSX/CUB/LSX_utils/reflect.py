import os
import sys
sys.path.append('NSFRAlpha/src/')
import torch
import shutil
import numpy as np

from tqdm import tqdm

from NSFRAlpha.src.nsfr_utils import get_nsfr_model
from NSFRAlpha.src.logic_utils import get_lang
from CUB.dataset import PROP_STR_GROUPED_FLAT


def reflect(lsx_iter, args, rr=False, topk=1, loader=None, verbose=0):
	assert topk > 0

	extracted_clauses = []
	preds = []

	for pos_classid in range(args.n_imgclasses):

		print(f"Class {pos_classid}, loading NSFR")

		# load logical representations
		lark_path = 'NSFRAlpha/src/lark/exp.lark'
		lang_base_path = os.path.join(args.log_dir, "proposed_clauses")
		lang, clauses, bk_clauses, bk, atoms = get_lang(
			lark_path, lang_base_path, dataset_type=f"CUB-{pos_classid}", dataset=None)

		C = len(clauses)
		print("Number eval clauses: ", C)
		# update infer module with new clauses# update infer module with new clauses
		logic_model = get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, args.device, train=False)

		score_pos = torch.zeros((C,)).to(args.device)
		score_neg = torch.zeros((C,)).to(args.device)
		score_pos_norm = torch.ones((C,)).to(args.device)
		score_neg_norm = torch.ones((C,)).to(args.device)
		if C > 1:
			N_data_pos = 0

			print(f"Scoring validation set for class {pos_classid} based on explanation extracted knowledge")

			# get only positive sample ids
			loader.dataset.update_pos_sample_ids(pos_classid)
			# get scores for positive samples
			for i, sample in tqdm(enumerate(loader, start=0)):
				inputs, labels = sample
				labels = labels.to(args.device)
				if isinstance(inputs, list):
					# inputs = [i.long() for i in inputs]
					inputs = torch.stack(inputs).t()  # .float()
				inputs = torch.flatten(inputs, start_dim=1).float().to(args.device)

				# print(logic_model.clauses)
				N_data_pos += inputs.size(0)
				B = inputs.size(0)
				# C * B * G
				V_T_list = logic_model.clause_eval(inputs).detach()
				C_score = torch.zeros((C, B)).to(args.device)
				for i, V_T in enumerate(V_T_list):
					predicted = logic_model.predict(v=V_T, predname='pos').detach()
					C_score[i] = predicted
				# sum over positive prob
				C_score = C_score.sum(dim=1)
				score_pos += C_score

			N_data_neg = 0

			# get only negative sample ids
			loader.dataset.update_neg_sample_ids(pos_classid)

			# get scores for negative samples
			for i, sample in tqdm(enumerate(loader, start=0)):
				inputs, labels = sample
				labels = labels.to(args.device)
				if isinstance(inputs, list):
					# inputs = [i.long() for i in inputs]
					inputs = torch.stack(inputs).t()  # .float()
				inputs = torch.flatten(inputs, start_dim=1).float().to(args.device)

				# print(logic_model.clauses)
				N_data_neg += inputs.size(0)
				B = inputs.size(0)
				# C * B * G
				V_T_list = logic_model.clause_eval(inputs).detach()
				C_score = torch.zeros((C, B)).to(args.device)
				for i, V_T in enumerate(V_T_list):
					predicted = logic_model.predict(v=V_T, predname='pos').detach()
					C_score[i] = predicted
				# sum over positive prob
				C_score = C_score.sum(dim=1)
				score_neg += C_score

			score_pos_norm = score_pos / N_data_pos
			score_neg_norm = score_neg / N_data_neg
			del inputs

		score_joint_pre_norm = score_pos_norm - score_neg_norm

		if verbose:
			for i in range(len(clauses)):
				print(
					f"{i}: joint:{score_joint_pre_norm[i]:.4f} pos:{score_pos_norm[i]:.4f} neg:{score_neg_norm[i]:.4f} {clauses[i]}")

			top_ids = torch.topk(score_joint_pre_norm, k=min(len(score_joint_pre_norm), 5))[1]
			print("Top-k:\n")
			for i in range(len(top_ids)):
				cur_id = top_ids[i]
				print(
					f"{clauses[cur_id]} \njoint:{score_joint_pre_norm[cur_id]} pos:{score_pos_norm[cur_id]} neg:{score_neg_norm[cur_id]}\n")

		# only take maximal probable clause
		if topk==1:
			class_list = []
			max_id = torch.argmax(score_joint_pre_norm)
			print(
				f"\nMax:{clauses[max_id]} \njoint:{score_joint_pre_norm[max_id]} pos:{score_pos_norm[max_id]} neg:{score_neg_norm[max_id]}\n")
			preds.append(f"cub{pos_classid}:1:image")
			clauses[max_id].head.pred.name = f"cub{pos_classid}"
			class_list.append(clauses[max_id])
			extracted_clauses.append(class_list)
		# otherwise take topk maximally probable clauses
		else:
			# cathc case if there are less proposed clauses than topk
			if len(score_joint_pre_norm) <= topk:
				top_ids = np.arange(0, len(score_joint_pre_norm))
			else:
				top_ids = torch.topk(score_joint_pre_norm, k=topk)[1]
			preds.append(f"cub{pos_classid}:1:image")
			class_list = []
			for i in range(len(top_ids)):
				cur_id = top_ids[i]
				print(
					f"{clauses[cur_id]} \njoint:{score_joint_pre_norm[cur_id]} pos:{score_pos_norm[cur_id]} neg:{score_neg_norm[cur_id]}\n")
				clauses[cur_id].head.pred.name = f"cub{pos_classid}"
				class_list.append(clauses[cur_id])
			extracted_clauses.append(class_list)

		del logic_model, score_pos, score_neg, score_pos_norm, score_neg_norm

	# set sample ids of val laoder back to full list
	loader.dataset.reset_sample_ids()

	extracted_clauses_flat = [item for sublist in extracted_clauses for item in sublist]
	save_to_txt(extracted_clauses_flat, preds, args)

	base_nsfr_lang_path = os.path.join('NSFRAlpha', 'data', 'lang', args.dataset_type)

	# load new extracted clauses and preds into logic model
	lang, clauses, bk_clauses, bk, atoms = get_lang(
		lark_path, lang_base_path, dataset_type="", dataset=None)
	args.m = len(clauses)
	# update infer module with new clauses
	logic_model = get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, args.device, train=False)

	# if just for rrr, return rules directly as list of strings
	if rr:
		clauses = [[rule.__str__() for rule in class_clauses] for class_clauses in extracted_clauses]
		# if topk == 1:
		# 	clauses = [rule.__str__() for rule in extracted_clauses]
		# else:
		# 	clauses = [[rule.__str__() for rule in class_clauses] for class_clauses in extracted_clauses]

	return logic_model, clauses


def save_to_txt(clauses, preds, args):
	save_dir = os.path.join(args.log_dir, "proposed_clauses")
	os.makedirs(save_dir, exist_ok=True)

	textfile = open(os.path.join(save_dir, f"clauses.txt"), "w")
	for rule in clauses:
		textfile.write(rule.__str__() + "\n")
	textfile.close()

	textfile = open(os.path.join(save_dir, f"preds.txt"), "w")
	for pred in preds:
		textfile.write(pred + "\n")
	textfile.close()

	# copy base lang into each class folder
	base_nsfr_lang_path = os.path.join('NSFRAlpha', 'data', 'lang', args.dataset_type)
	if not os.path.isfile(os.path.join(save_dir, 'neural_preds.txt')):
		shutil.copyfile(os.path.join(base_nsfr_lang_path, 'neural_preds.txt'),
		                os.path.join(save_dir, 'neural_preds.txt'))
		shutil.copyfile(os.path.join(base_nsfr_lang_path, 'consts.txt'),
		                os.path.join(save_dir, 'consts.txt'))


if __name__ == "__main__":
	main()
