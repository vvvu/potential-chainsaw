### Summary

#### PyTorch Basics

1. `requires_grad = True/False`可以用来表示是否需要在当前计算中保留对应的梯度信息，这可以让我们手动控制将梯度回传控制在网络中的某个位置：只需要将对应结点的`requires_grad = False`即可，此时梯度无法回传，**这样我们就可以针对该层进行调参**
2. 自制数据集`How to build custom dataset`和使用预训练模型`Using pretrained model`
3. `device`选择设备
   1. `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`配置当前设备为GPU/CPU
   2. `.to(device)`可以将对应的Tensor操作安置到我们之前定义的`device`中，方便我们在不同的GPU/CPU设备中运算

#### Regression

1. 优化器以及BP相关

   - `optimizer.zero_grad()`可以将计算图的梯度清零，在每次进行BP计算之前我们都需要将梯度清零

   - `loss.backward()`即BP过程

   - `optimizer.step()`即梯度下降更新参数过程

2. `with torch.no_grad():`可以强制之后的内容不进行计算图的构建，从而节省内存资源。**这是因为PyTorch默认会将所有的运算操作都加入到计算图中，对于Tensor的计算操作，也是会加入到计算图中。但实际上，在Training阶段我们需要这样做，但是在Test阶段我们无需构建计算图，所以我们通过`with torch.no_grad()`来强制之后的内容不进行计算图的构建**
