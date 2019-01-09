1.  Model, loss function and cost function for binary classification

   ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_basic\FireShot Capture 1 - Logistic Regression Cost Function - de_ - https___www.coursera.org_learn_neur.png)


model:  是一个线性方程套上sigmod方程的形式，保持了y预测值在0~1之间（含义是取y=1的概率）

loss：loss针对每一个样本，用(y'-y)^2的方式定义的loss会有局部最优的问题，所以该用了另外一种方式（这就是所谓的交叉熵的方式）
$$
Loss(y',y) = -(ylogy' + (1-y)log(1-y'))
$$
对于一个2项分布，如果认为存在以下公式，用(y,y')来表示p(y|x)，即可由最大似然的方式，得到单个loss func 以及整体cost func的表达
$$
P(y|x) = y'^y * (1-y')^(1-y)
$$
即，因为log的单调性，使得P(y|x)最大，就是使log(P(y|x))最大，也就是Loss(y‘,y)最小

使全体达到最大似然效果，就是使m个P连乘最大，因为log的单调性结果，所以变成了m个-loss的和最大，即变成优化m个loss的合计最小，这就是cost func的由来

cost function： 优化时最终需要保证整体训练集表现最优，所以通过定义cost（各个lost的某种综合体现）的方式定义，这里选用了各项的均值



2. Derivatives

![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_basic\FireShot Capture 4 - Logistic Regression Gradient Descent -_ - https___www.coursera.org_learn_neur.png)

一个最简单的使用链式法则推到，通过逐层转化，最终得到dw1，dw2等的值（w1，w2相对于Loss的导数），以实现优化迭代。

在实际使用时，要整体优化，即计算m个样本的结果。

需要注意整体优化时，不是用for循环，而是直接用向量计算实现乘算的，np的向量计算可以大幅优化计算速度（某个例子里达到了for循环的300倍不止，这也是为什么使用GPU运算的核心）

```python
import numpy as np
# np compute
result = np.dot(w,X) # 点乘
result = np.exp(x) # 对向量x，计算新向量，对应xi得值是math.exp(xi)
result = np.log(x) # 同上，换做对数形式
result = np.abs(x) # absolute
result = np.maximum(x,0) # 对xi, 取xi与0中较大的值(ReLu)
# np broadcasting
np.arrays([[1,2,3,4,5],[6,7,8,9,0]]) + 1 
# also not use
a = np.random.randn(5) # (5,) not a typical vector
a = np.random.randn(5,1) # (5,1) vector, use this instead
# 
```

使用Vectorization代替for循环也是很多优化的核心（使用numpy的built-in代替）

以下是一个全面替代后的形式：

![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_basic\FireShot Capture 5 - Vectorizing Logistic Regression's Grad_ - https___www.coursera.org_learn_neur.png)



作为一次训练，左边是for loop写成的形式，右边是用Vectorization代替的形式。

**注意无论是那边，都是先正向求算出当前状态下的预测值，再反向求出导数，随后做更新的。**

也就是先正向传播（预测过程），再反向传播（求导与更新过程）

在右侧，由于Vectorization的介入，使得之前m * n的循环被避免了（这里整理一下各个向量的形状）

```python
'''
m:样本数
n:特征数
X : (n, m)
w : (n, 1)
b : 这里用常数代替，实际使用的时候被转化成了(1,m)
Z : (1, m)
A : (1, m)
dZ : (1, m)
dw : (n, 1) #  与w相同
db : (1, 1)
'''
```

右边最外层的循环是为了实现多次训练的，对于每一次训练，都是对m个样本的loss合计（cost）做了一次优化，使得优化后的w,b和之前相较产生更小的cost结果



3 .  multiple units

![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_basic\FireShot Capture 9 - Computing a Neural Network's Output - _ - https___www.coursera.org_learn_neur.png)

Using vector and matrix to represent the Z-equation and a-equation

各层之间有着相似的运算逻辑，上图只是对一个样本进行了运算，通过横向堆叠，向前传播的过程可以对m个样本同步（向量化）完成。

同时注意参数只有一组，**以及这种对Z，a（x）横向表示不同样本，纵向表示不同特征(节点)的写法**

![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_basic\FireShot Capture 10 - Vectorizing across multiple examples -_ - https___www.coursera.org_learn_neu.png)



4 . activition function

![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_basic\FireShot Capture 11 - Activation functions - deeplearning.ai_ - https___www.coursera.org_learn_neu.png)

四种激励函数的选择，因为tanh在各种意义上优于sigmoid,(-1,1)的区间使得激活后的均值趋向0，所以导致sigmoid几乎不用于实践，而ReLU计算的便利性，以及对于函数远端难下降问题的解决成为绝大多数模型的首选。

当然，sigmoid可以被用作binary问题的最后一层，或者将其理解成一个只有单位是1的softmax

为什么一定要有一个non-linear的activation function，为什么不直接用a(z) = z （尽管他和relu已经很像了）作为activation functions?

![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_basic\FireShot Capture 12 - Why do you need non-linear activation _ - https___www.coursera.org_learn_neu.png)

因为这样一来，线性自由叠加的特性会让所有相邻的线性神经元（各层，或者同层的各神经元）整合成同一个线性方程而失去其原先的结构。

所以，除去最后一个层以计算y的神经元（out-put layer）外（当出现预测实数，而非分类问题时，可以视情况让最后一层使用Relu或者linear），中间的hidden-layer使用linear的activation function是没有意义的



5. forward propagation and backward propgation

![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_basic\FireShot Capture 16 - Gradient descent for Neural Networks -_ - https___www.coursera.org_learn_neu.png)

结合单元2的图2，可以更明确的得到上面结果，即先通过向前传播，在上一组参数和输入X的基础上计算出A[output]，再反向逐层计算导数，最终得到每一层参数的修正值。

**这里因为默认最后一层使用sigmoid得到binary的Y，同时loss函数写成单元1中形式，所以loss函数对Z[output]的导数被写成了A[output]-Y**

np.sum(..., keepdim=True)是为了保证python不产生1-rank数据的

A full version of derivatives with samples

![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_basic\FireShot Capture 17 - Backpropagation intuition (optional) I_ - https___www.coursera.org_learn_neu.png)

(with these two, I can write a full connection simple neural network without the help of tensor-flow)

6. Why we need to initialize weight randomly

Then the net work will compute out a bunch of 0's in forward propagation;

Then they will update to the same values in backward propagation

As a result, all units become identical with computing same functions

typical initial:

w = np.random.randn((n,n))*0.01 # leave this small to make it easy to compute gradient(learn faster) for sigmoid( caution the output layers could be sigmoid/softmax)

b = np.zeros((n,1))

