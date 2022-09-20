# EfficientFormers

**EfficientFormers** is a library which contains some SOTA efficient transformer architectures. Be note that, effcientformers only index **light-weighted and fast** transformer models. 

Currently supported models:

- [efficientformer - 2022](https://github.com/snap-research/EfficientFormer);


will add:

- [EcoFormer - 2022](https://github.com/ziplab/EcoFormer);


Please star and fork! Contribution are very welcomed.



## Install

```
pip install efficientformers
```


## Usage

```python

from efficientformers.cv.efficientformer import efficientformer_l1
import torch


x = torch.randn([1, 3, 224, 224])
a = efficientformer_l1(num_classes=10, distillation=False)
o = a(x)

print(o.shape)

```