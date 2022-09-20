from efficientformers.cv.efficientformer import efficientformer_l1
import torch


x = torch.randn([1, 3, 224, 224])
a = efficientformer_l1(num_classes=10, distillation=False)
o = a(x)

print(o.shape)

