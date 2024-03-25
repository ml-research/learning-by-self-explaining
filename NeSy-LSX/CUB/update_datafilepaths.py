import pickle
import argparse
import os


def parse_arguments():

	parser = argparse.ArgumentParser(description='CUB Training')
	parser.add_argument('--path_to_CUB', type=str, help='Filepath to CUB_200_2011')
	parser.add_argument('--path_to_processed', type=str, help='filtered dataset files')
	args = parser.parse_args()
	return args


def update_fps(args):
	for data_fname in ['train.pkl', 'test.pkl', 'val.pkl']:
		data_fpath = os.path.join(args.path_to_processed, data_fname)
		with open(data_fpath, 'rb') as f:
			data = pickle.load(f)

		data_orig = data.copy()
		for i in range(len(data)):
			data[i]['img_path'] = os.path.join(args.path_to_CUB, data[i]['img_path'].split('CUB_200_2011'+os.path.sep)[-1])

		with open(data_fpath, 'wb') as handle:
			pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print(f"updated {data_fpath}")

if __name__ == '__main__':

	args = parse_arguments()
	update_fps(args)
