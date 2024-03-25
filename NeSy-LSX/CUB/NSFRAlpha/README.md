# alpha-ILP: Differentiable Learning of Logical Representations from Visual Scenes
This is the implementation of alpha-ILP.

![ailp](./imgs/aILP.png)

## Installation
The packages are specified in [requirements.txt](./requirements.txt). Please install the packages by:
```
pip install -r requirements.txt
```

## Training
### Kandinsky Patterns Twopairs Dataset:
```
python src/train.py --dataset-type kandinsky --dataset twopairs --batch-size 24 --no-cuda --n-beam 5 --t-beam 5
```
### CLEVR-Hans3 Dataset:
```
python src/train.py --dataset-type clevr --dataset clevr-hans0 --batch-size 24 --no-cuda --n-beam 5 --t-beam 5
```


![twopairs](./imgs/twopairs-predicted.png)
![clevr-hans3](./imgs/clevr-hans3-predicted.png)
![clevr-hans7](./imgs/clevr-hans7-predicted.png)




# LICENSE
See [LICENSE](./LICENSE). The [src/yolov5](./src/yolov5) folder is following [GPL3](./src/yolov5/LICENSE) license.