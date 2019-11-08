---
layout: post
title: Pytorch Program Note
mathjax: true
categories: Program
tags:
keywords:
description: Some Program Tips
mermaid: true
status: Writing
---

> F - torch.nn.functional

# F.grid_sample
xx

# 计算文件夹内文件数量

```python
def visitDir(path):
    if not os.path.isdir(path):
        print('Error: "', path, '" is not a directory or does not exist.')
        return
    else:
        global x
        try:
            for lists in os.listdir(path):
                sub_path = os.path.join(path, lists)
                x += 1
#                 print('No.', x, ' ', sub_path)
                if os.path.isdir(sub_path):
                    visitDir(sub_path)
        except:
            pass

x = 0
visitDir(args.img_root)
print(x)
```

# contiguous()

view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。 

```python
import torch
x = torch.ones(10, 10)
x.is_contiguous()  # True
x.transpose(0, 1).is_contiguous()  # False
x.transpose(0, 1).contiguous().is_contiguous()  # True
```

# masked_fill_

给一个 tensor， 将 mask 对应为 1 的位置使用固定的数代替。

```python
aa = torch.ByteTensor ([1,1,0,1,1,0,1])
cc = torch.rand(aa.shape)

cc.data.data.masked_fill_(aa.data, -float('inf'))

# output: tensor([  -inf,   -inf, 0.1134,   -inf,   -inf, 0.4796,   -inf])
```

