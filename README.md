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

```
Goto src folder
cd src
```

```
CIFAR-10
```

```
python train.py -b 16 -lr 0.001 -e 50 -d cifar10
```

```
python test.py -b 16 -lr 0.001 -e 50 -d cifar10
```

```
Experimental Model
```

```
python train_exp.py -b 16 -lr 0.001 -e 50 -d cifar10
```

```
python test_expr.py -e 50 -d cifar10
```

```
CIFAR-100
```

```
python train.py -b 16 -lr 0.001 -e 50 -d cifar100
```

```
python test.py -b 16 -lr 0.001 -e 50 -d cifar100
```

```
Experimental Model
```

```
python train_exp.py -b 16 -lr 0.001 -e 50 -d cifar100
```

```
python test_expr.py -e 50 -d cifar100
```





