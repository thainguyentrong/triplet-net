# Triplet Network
Pytorch implementation of Triplet Network for Content-based Image Retrieval (CBIR).
## Dataset
We use the [Oxford dataset](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) and [Paris dataset](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) to verify the Triplet Network. The processed datasets can be downloaded [here](https://drive.google.com/file/d/1mrfchgX167GZZ4Wbk5ig8lcDRQy2eXsD/view?usp=sharing)
## How to train
* Check ```requirements.txt``` before using repos
* Check ```config.py``` containing the network configuration
* Download the dataset, extract and move it to ```./dataset```
* Run ```python train.py```
## Evalute
* Run ```python demo.py```
## Pretrained model
* You can download pretrained model in here: [resnet34 with Oxford5k](https://drive.google.com/file/d/1902WjolDw0Ch5qzTSzL8gbpqtGUjNYzn/view?usp=sharing) and [resnet34 with Paris](https://drive.google.com/file/d/1-DAbCuRwYQl_RELIcofmrc3b1VI5YZaa/view?usp=sharing)
* Move it to ```./model/resnet/```
* Run ```python train.py```
