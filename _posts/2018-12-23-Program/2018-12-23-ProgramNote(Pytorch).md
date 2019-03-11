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
