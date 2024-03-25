# CNN-LSX instantiation

## ---Running code---

E.g. for training CNN-LSX via a small percentage of mnist on GPU 0, with seed 0 and with 
lambdac = 1, lambdae = 100 and lambdae_f = 100: 

```
./scripts/train_net1_lsx_smalldata.sh 0 0 mnist 1 100 100
```

Please see ```utils.py``` for the different dataset configurations, e.g. unconfounding.
```scripts/``` contains example bash scripts for different experiments and analysis, take a look there as well. 

```parameters/``` contains a few example command line arguments for individual experiments, in case 
the bash scripts are not clear.

Be aware it might take some epochs until you see the explanation loss decreasing. In my experience there is a 
form of "breaking point" after which the optimization process focuses mainly on optimizing the explanations. To speed 
things up you can also consider setting lambdae very high, e.g. 1000. This leads to the classification loss during 
joint optimization being nearly neglected, but in practice this is fine, as this can be optimized again more easily
when we finetune in the ```n_finetuning_epochs```, e.g. after the ```n_epochs``` (we refer to the paper's Suppl. for 
details on this).  

**Warning:** Note the structure of this repository is by no means good. There are unnecessary arguments, overly 
complex class structures etc. Sorry for this ... I might suggest to rewrite your own version of this based on our code.  

## ---Additional Data Information---

For obtaining the colormnist dataset we recommend looking into this [repo](https://github.com/ml-research/NeSyXIL) 
(see bottom of that README)
