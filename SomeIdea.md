### RAPF和Libcontinual

1. RAPF的模型参数是由一个clip包的预训练模型和一个线性适配器（称adapter，就是一层nn.Linear）组成的。读论文发现，这个算法是靠调整adapter的参数来达到对预训练模型作微调训练相同的效果。
2. 在Libcontinual里添加算法需要我们写一个继承了它给的finetune类的模型，包含初始化等几种方法。这个类有backbone, feat_dim, num_class, classifier, loss_fn等等属性。在`process.md`里可以看到Libcontinual用`trainer.py`循环调用一个模型的各种方法来完成训练。
3. 但是RAPF没有使用框架里的任何一种backbone，调整的参数也仅有一层adapter，我不知道应该怎么让这个结构融入框架。目前我只是简单粗暴地把更新权重都放进after_task方法。

### clip_increamental.py已经做的事情

我对照着RAPF的`model.py`和`main.py`把init, before_task, observe, after_task都整合上了。没有做get_parameters。因为我这种做法让adapter的参数不会跟着trainer的_train方法训练，而是在任务后处理中更新。

### 可能的思路

或许可以把adapter和clip预训练模型拿出来重写一个backbone放进backbone文件夹，这样就符合框架了。但是由于RAPF更新权重的方式特殊，依然需要在任务后处理将权重改变，然后在下一轮训练的任务前处理用get_parameters调出来。