# cs536-ml-final-mobilenetv2

### Team Members
|NAME|BNUMBER|EMAIL|USERNAME|
--- | --- | --- |---
Apoorv Yadav|B00977654|ayadav7@binghamton.edu |ayadav16
Bhavit Yogesh Shah|B00979233|bshah5@binghamton.edu|bhavityogeshshah

### Contribution by each member:

Both team member have contributed equally and done their part of the project. They have decided on the experiment together and wrote the code for it. Please refer git log history for exact contribution for each member. 

### Steps to Run:
Please contact with the team incase the code fails to run due to dependency error.


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





