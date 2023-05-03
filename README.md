# cs536-ml-final-mobilenetv2

### Team Members
|NAME|BNUMBER|EMAIL|
--- | --- | --- 
Apoorv Yadav|B00977654|ayadav7@binghamton.edu 
Bhavit Yogesh Shah|B00979233|bshah5@binghamton.edu


```
git clone https://github.com/bhavityogeshshah/cs536-ml-final-mobilenetv2.git
```

```
pip3 -r install requirements.txt
```


Goto src folder
```
cd src
```


## Original CIFAR-10

Train:
```
python train.py -b 16 -lr 0.001 -e 50 -d cifar10
```
Test:
```
python test.py -b 16 -lr 0.001 -e 50 -d cifar10
```


## Experimental Model

Train:
```
python train_exp.py -b 16 -lr 0.001 -e 50 -d cifar10
```
Test:
```
python test_expr.py -e 50 -d cifar10
```


## Original CIFAR-100

Train:
```
python train.py -b 16 -lr 0.001 -e 50 -d cifar100
```
Test:
```
python test.py -b 16 -lr 0.001 -e 50 -d cifar100
```


## Experimental Model

Train:
```
python train_exp.py -b 16 -lr 0.001 -e 50 -d cifar100
```
Test:
```
python test_expr.py -e 50 -d cifar100
```





