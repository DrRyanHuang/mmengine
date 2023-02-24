# 模型复杂度分析

We provide a tool to help with the complexity analysis for the network. We borrow the idea from the implementation of [fvcore](https://github.com/facebookresearch/fvcore) to build this tool, and plan to support more custom operators in the future. Currently, it provides the interfaces to compute "parameter", "activation" and "flops" of the given model, and supports printing the related information layer-by-layer in terms of network structure or table. The analysis tool provides both operator-level and module-level flop counts simultaneously. Please refer to [Flop Count](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md) for implementation details of how to accurately measure the flops of one operator if interested.

## 何为 FLOP

FLOP 是 **Fl**oating **P**oint **OP**eration 的简称，也就是一次浮点运算操作，但其在模型复杂度分析中没有一个明确的定义。`mmengine` 与 [detectron2](https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.nn.FlopCountAnalysis) 使用相同的统计方式, 将一次混合的乘加操作视为一次 FLOP

## 何为 Activation

Activation is used to measure the feature quantity produced from one layer.

For example, given the inputs with shape `inputs = torch.randn((1, 3, 10, 10))`, and one linear layer with `conv = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=1)`.

We get the `output` with shape `(1, 10, 10, 10)` after feeding the `inputs` into `conv`. The activation quantity of `output` of this `conv` layer is `1000=10*10*10`

Let's start with the following examples.


Activation is used to measure the feature quantity produced from one layer.

比如, 给定输入变量 `inputs = torch.randn((1, 3, 10, 10))` 和一个卷积层模型 `conv = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=1)`。

前向传播之后，得到形状为 `(1, 10, 10, 10)` 输出值 `output`。

那么, 这个卷积层模型输出的 activation 值为 `1000=10*10*10`。

接下来, 通过两个例子来看 `mmengine` 如何进行模型复杂度分析。

## 使用示例 1: 当使用 nn.Module 构建模型时

### 代码

```python
import torch
from torch import nn
from mmengine.analysis import get_model_complexity_info
# return a dict of analysis results, including:
# ['flops', 'flops_str', 'activations', 'activations_str', 'params', 'params_str', 'out_table', 'out_arch']

class InnerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10,10)
        self.fc2 = nn.Linear(10,10)
    def forward(self, x):
        return self.fc1(self.fc2(x))


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10,10)
        self.fc2 = nn.Linear(10,10)
        self.inner = InnerNet()
    def forward(self, x):
        return self.fc1(self.fc2(self.inner(x)))

input_shape = (1, 10)
model = TestNet()

analysis_results = get_model_complexity_info(model, input_shape)

print(analysis_results['out_table'])
print(analysis_results['out_arch'])

print("Model Flops:{}".format(analysis_results['flops_str']))
print("Model Parameters:{}".format(analysis_results['params_str']))
```

### 结果描述

The return outputs is dict, which contains the following keys:

- `flops`: number of total flops, e.g., 10000, 10000
- `flops_str`: with formatted string, e.g., 1.0G, 100M
- `params`: number of total parameters, e.g., 10000, 10000
- `params_str`: with formatted string, e.g., 1.0G, 100M
- `activations`: number of total activations, e.g., 10000, 10000
- `activations_str`: with formatted string, e.g., 1.0G, 100M
- `out_table`: print related information by table

```
+---------------------+----------------------+--------+--------------+
| module              | #parameters or shape | #flops | #activations |
+---------------------+----------------------+--------+--------------+
| model               | 0.44K                | 0.4K   | 40           |
|  fc1                |  0.11K               |  100   |  10          |
|   fc1.weight        |   (10, 10)           |        |              |
|   fc1.bias          |   (10,)              |        |              |
|  fc2                |  0.11K               |  100   |  10          |
|   fc2.weight        |   (10, 10)           |        |              |
|   fc2.bias          |   (10,)              |        |              |
|  inner              |  0.22K               |  0.2K  |  20          |
|   inner.fc1         |   0.11K              |   100  |   10         |
|    inner.fc1.weight |    (10, 10)          |        |              |
|    inner.fc1.bias   |    (10,)             |        |              |
|   inner.fc2         |   0.11K              |   100  |   10         |
|    inner.fc2.weight |    (10, 10)          |        |              |
|    inner.fc2.bias   |    (10,)             |        |              |
+---------------------+----------------------+--------+--------------+
```

- `out_arch`: print related information by network layers

```bash
TestNet(
  #params: 0.44K, #flops: 0.4K, #acts: 40
  (fc1): Linear(
    in_features=10, out_features=10, bias=True
    #params: 0.11K, #flops: 100, #acts: 10
  )
  (fc2): Linear(
    in_features=10, out_features=10, bias=True
    #params: 0.11K, #flops: 100, #acts: 10
  )
  (inner): InnerNet(
    #params: 0.22K, #flops: 0.2K, #acts: 20
    (fc1): Linear(
      in_features=10, out_features=10, bias=True
      #params: 0.11K, #flops: 100, #acts: 10
    )
    (fc2): Linear(
      in_features=10, out_features=10, bias=True
      #params: 0.11K, #flops: 100, #acts: 10
    )
  )
)
```

## 使用示例 2: 当使用 mmengine 构建模型时

### 代码

```python
import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel
from mmengine.analysis import get_model_complexity_info


class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels=None, mode='tensor'):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels
        elif mode == 'tensor':
            return x


input_shape = (3, 224, 224)
model = MMResNet50()

analysis_results = get_model_complexity_info(model, input_shape)


print("Model Flops:{}".format(analysis_results['flops_str']))
print("Model Parameters:{}".format(analysis_results['params_str']))
```

### 输出

```bash
Model Flops:4.145G
Model Parameters:25.557M
```

## 用户接口

`mmengine` 提供了一些选项来支持自定义输出

- `model`: (nn.Module) the model to be analyzed
- `input_shape`: (tuple) 输入的形状, 例如, (3, 224, 224)
- `inputs`: (optional: torch.Tensor), 若给定, 参数 `input_shape` 将被忽略
- `show_table`: (bool) whether return the statistics in the form of table, 默认为: True
- `show_arch`: (bool) whether return the statistics in the form of table,  默认为: True
