# NeSy-LSX for CUB-10

## ---Data prepreparation---
This repository focuses on NeSy-LSX for the CUB-10 dataset. This is very much based on the repository of Koh et al. 2020
on (CBMs)[https://github.com/yewsiang/ConceptBottleneck]. The original CUB dataset can be downloaded here: 
[Caltech-UCSD Birds 200 (CUB)](http://www.vision.caltech.edu/visipedia/CUB-200.html).

We use the sequential bottleneck case, i.e. we pretrain the bottleneck CNN model based on the original code 
and preprocess the concept labels based on our ```data_processing.py``` (see ```CUB/scripts/experiments.sh``` 
for details on these steps). ```CUB_processed/``` already contains our "majority with noise" preprocessed 
data.

## ---Running Code---
```CUB/scripts/experiments_lsx.sh``` provides example script calls.

```parameters/``` contains additional log files containing evaluation arguments.
