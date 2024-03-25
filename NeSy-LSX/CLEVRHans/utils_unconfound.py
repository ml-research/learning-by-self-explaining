import numpy as np

import data_clevr_hans as data


def update_val_and_test_dataset(loaders, n_samples_per_class, args):
	"""
	Take several samples from the unconfounded test set and make these the validation set,
	remove these from the test set
	"""

	# create unconfounded test set
	dataset_test = data.CLEVR_HANS_EXPL(
		args.data_dir, "test", lexi=True, conf_vers=args.conf_version
    )
	dataset_val = data.CLEVR_HANS_EXPL(
		args.data_dir, "test", lexi=True, conf_vers=args.conf_version
	)
	test_loader = data.get_loader(
		dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
	)
	val_loader = data.get_loader(
		dataset_val, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False
	)

	test_sample_ids = np.array(test_loader.dataset.sample_ids)
	test_imgs_class_ids = np.array(test_loader.dataset.img_class_ids)

	new_sample_ids = []
	for class_id in range(args.n_imgclasses):
		rel_ids = np.where(test_imgs_class_ids == class_id)[0][:n_samples_per_class]
		new_sample_ids.extend(test_sample_ids[rel_ids])

	new_sample_ids = list(np.random.permutation(np.array(new_sample_ids)))

	# update validation dataset
	# val_loader_new = data.get_loader(
	# 	loaders[2].dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False
	# )
	val_loader.dataset.full_sample_ids = new_sample_ids
	val_loader.dataset.sample_ids = val_loader.dataset.full_sample_ids.copy()

	# remove the sample ids from the test set that were just chosen for the validation set
	test_loader.dataset.full_sample_ids = list(set(test_loader.dataset.sample_ids) - set(new_sample_ids))
	test_loader.dataset.sample_ids = test_loader.dataset.full_sample_ids.copy()

	return (loaders[0], val_loader, test_loader)