# NeSy-LSX for CLEVR-Hans3

## ---Data prepreparation---
This repository focuses on LSX for the CLEVR-Hans3 dataset. In the normal classification settings in 
our evaluations (i.e. not the confounding evaluations) we use the original validation set as held-out test 
and select a subset of the training data as validation set. These are the steps to load and preprocess
the data:

1. Follow the instructions of the (CLEVR-Hans)[https://github.com/ml-research/CLEVR-Hans] repository for downloading
the original dataset
2. Follow the ```create_clevrhans_splits.ipynb``` jupyter notebook for creating the new train and validation splits

## ---Running Code---
E.g. for training CNN-LSX via a small percentage of mnist on GPU 0, with seed 0, run number 0 and datapath

```
./scripts/train_alldata.sh 0 0 0 /pathto/CLEVR-Hans3/
```

Please see ```utils.py``` for the different dataset configurations, e.g. unconfounding.
```scripts/``` contains example bash scripts for different experiments and analysis, take a look there as well. 

```parameters/``` contains a few example command line arguments for individual experiments, in case 
the bash scripts are not clear.

Hint: The propositionalisation and reflect step via the NSFR critic module can take a bit of time and potentially can 
lead to memory issues. You might have to tweak with the ```prop-thresh``` argument (see utils for argument
description and ```propositionalise()``` in ```utils_reflect.py```) to reduce the number of candidate explanations.  

Hint2: I never really knew why, but sometimes, the code would get stuck in the propositionalising step in this case 
just restarting the script did not run in the same problem. Wacky stuff ...
