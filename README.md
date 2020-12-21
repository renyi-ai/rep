# Setup

## Install

You will need pytorch and torchvision installed.

```bash
git clone https://github.com/renyi-ai/rep.git
cd rep
git checkout pgergo
```

## Downloads
You will need to store your data with the following folder structure:
```bash
.
├── res
│   ├── cifar10
│      ├── data
│           ├── cifar-10-batches-py
│               ├── batches.meta
│               ├── data_batch_1
│               ├── data_batch_2
│               ├── ...
│       ├── models
│           ├── densenet121.pt
│           ├── densenet161.pt
│           ├── ...
│       
```

In order to do that, download models and data as below.

### Models
This will download models to res/cifar10/models/
```bash
python src/bin/download.py
```

### Data
By making your first run, the data_loader will look into the res/cifar10/data folder
and if it founds the cifar-10-batches-py directory there, it continues with that. If not,
it will download it.

## Running

In order to analyse how a perturbation modifies the model's behaviour, you need to define two functions.

### Manipulation
The first function needs to be defined is under src/functions/manipulator.py. The function you define here is going to be the modification you wish to apply in the middle of the network, on which ever layer you choose.

```python
# Example: Multiply each activation by 2
def twice(self, x):
    return 2*x
```
### Comparison

The second function needs to be defined is under src/functions/comparison.py. In order to evaluate your manipulation function, you will need something that represents the value of change. Here you can compare your modified logits' results with the original network's results, moreover, you can compare it with the original labels.

```python
# Example: accuracy for the model which has NOT been modified
def pre_acc(self, true_y, pre_y, post_y):
    # Shapes:
    # true_y : (10000,) (e.g. [7, 3, 2, ...])
    # pre_y  : (10000, 10) (one-hot)
    # post_y : (10000, 10) (one-hot)
    pred = pre_y.argmax(axis=1).flatten()
    return (pred==true_y).sum() / true_y.size
```

### Run analysis

You can choose from the following settings:
   - Which model to run (e.g. vgg16_bn)
   - Which layer index you want to modify (e.g. 20)
   - Your manipulation function (e.g. twice)
   - Your comaprison function (e.g. pre_acc)


```bash
python src/bin/perturbation_impact.py vgg16_bn 20 twice pre_acc
```