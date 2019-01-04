# 文献评论 ： Deep learning

### 文献摘要

​	该领域引用率最高的综述类文章其一，用相对简单的表述，介绍了DeepLearning的兴起，模型依据，及其应用最广泛的两大分支：图像识别，自然语言处理。



### 定位

​	文献结构的开端，我们的文献结构是由此篇DeepLearning开始，引出他讨论到的关键实验，再引申到和我们相关的（成像数据分析相关的）应用类文章上来的。希望从未接触过机器学习的读者，也能通过看这篇文章，掌握其中的核心概念与应用方式。



### 文章结构

​	文章共8个大分段，除去开头的引入和尾声对于未来的简单展望外，基本可以分解成：

| 段落  | 内容介绍                                                     |
| ----- | ------------------------------------------------------------ |
| 1,2,3 | 机器学习发展历史，相关基本原理介绍，其中遇到的问题，及deeplearning的优越性 |
| 4,5   | 卷积神经网络（CNN）， 与其在图像识别领域中的经典应用         |
| 6,7   | 循环神经网络（RNN），与其在自然语言处理中的应用              |
| 8     | 尾声                                                         |



### 内容评论

#### 段落1 ： 前言

​	前言部分有两个重要的信息，第一是在开始第2/3自然段，文章比较了传统的机器学习（Conventional）和Representation-learning的主要差别，另一点是在第一大段结尾的时候列举了其在各个领域最为由影响力的应用。

​	关于Conventional 和 Representation-learning的区别。

​	Conventional 就是最初被大家所熟知的各种机器学习算法，简单的包括线性回归，复杂一点如因子分析(factor analysis)， 支持向量机（SVM），马尔科夫决策模型（MDP）等。作者在文中，提到了传统机器学习的一个巨大缺陷，即需要训练者具有丰富的经验，让数据整顿到可以被对应模型抽象提取的状态。

​	而Representation-learning则极大程度上避免了这一点，其中deep-leaning（也就是全文所指的神经网络学习，不管是CNN，还是RNN）正式这种新的学习方法的重要代表。

```python
'''
Conventional machine-learning techniques were limited in their ability to process 
natural data in their raw form. For decades, constructing a pattern-recognition or 
machine-learning system required careful engineering and considerable domain expertise 
to design a feature extractor that transformed the raw data (such as the pixel values 
of an image) into a suitable internal representation or feature vector from which the 
learning subsystem, often a classifier, could detect or classify patterns in the input.

(前言，第二段中)

Representation learning is a set of methods that allows a machine to be fed with raw 
data and to automatically discover the representations needed for detection or 
classification. Deep-learning methods are representation-learning methods with multiple
levels of representation, obtained by composing simple but non-linear modules that each 
transform the representation at one level (starting with the raw input) into a 
representation at a higher, slightly more abstract level.

(前言，第三段首)
'''
```

​	用数字识别的例子做一个通俗的理解，如果所有的数字都是机器写出的标准字体，并且被等大小的放在每张图片的正中心，那么传统的学习手段也会表现的一样好（特别SVM在这种规律明显而非线性分割的情况下，可以用很少的样本便实现准确分割，这个可以参考SVM分割兰花的例子，参考链接：https://scikit-learn.org/stable/auto_examples/svm/plot_svm_nonlinear.html#sphx-glr-auto-examples-svm-plot-svm-nonlinear-py）

​	但是反过来说，如果大家不是标准字体，但是是相似的手写体，切不过其他位置条件满足，那么识别起来就相对麻烦一些（你可以在刚才的网页上找到SVM做数字识别的例子），如果这些字体即不是标准字体，相似程度又低，还歪七扭八的排布在图片的各个位置，那么使用SVM等传统方法，就很难了。除非，可以通过某些图像处理技术（例如通过边缘检测等找到数字位置，通过数字的水平竖直变等把数字正过来），让数字还原到之前可以被识别的状态。

​	上面提到的这种，先处理图像，在应用SVM（机器学习）的方法，就是文中所指的，在遇到复杂问题时，传统方法往往对于预处理的要求很高，而这延缓了效率，提高了门槛，切因为经验程度不同导致常常会有达不到效果的情况。

​	而Representation-learning，或者说本文提到的Deep-learning，则是一种自动的抽象过程，并以此规避掉了这一点，在实际的Deep-learning训练图片识别的过程中，输入的数字图片大小，方向，朝向各异，但是神经网络算法（deep-learning）通过一个个小的神经元对于图片的相似性的归类，提取，优化，可以在很短的内，通过一个不大的训练集（50000张28*28的图片），就达到了90%+的准确率的效果。且其中最便捷的一点是，用来训练的这个模型只是一个单纯的3层模型，除了定义层数，定义每层的神经元数量外，不需要额外的图像处理了。

​	且这个定义好的，三层，每层1000~2000的神经元的网络，并不局限于训练数字识别，其他的简单的特性识别，例如手写的标点符号，英文字母，等等几乎都是可以用同一个模型不加任何修改达到同样好的效果的。

​	这也正是**Deep-learning和核心，即“自动抽象”， 在学习过程计算机提取出的特征不是由有经验的工程师制定的，而是由机器根据数据自主归纳产生的**,。虽然在绝大部分被证明有用的模型中，开发者都无法准确说明为什么自己的神经网络学习可以实现这样的效果，但是从工业化的角度来说，因为他的这种自动抽象的特性，导致很多经过简单设计的模型已经可以起到之前复杂的传统模型无法实现的效果。**这也带来对于我们今天要做的，使用deep-learning重新挖掘脑成像数据的佐证，有别于之前传统的通过对齐，剪裁，提取，进行统计模型比较（比如fmri里面的计算HRF然后比较相关性得到contrast值），通过设计将成像数据喂（feed）给deep-learning系统后，他会自主抽象出可能存在的，用以区分不同刺激的特征以及组合方法。**

```python
'''
The key aspect of deep learning is that these layers of features are not designed by 
human engineers: they are learned from data using a general-purpose learning procedure.
（前言，第三段尾）
'''
```

​	在前言的后半部分，文章讲了deep-learning模型，在各个领域的应用，实际上，这里他主要提到的只有两个应用类型，一个是图像处理，一个是自然语言处理，分别对应了CNN和RNN网络，具体在应用中是面孔识别还是动物识别，是读文章写中心思想还是自动翻译，则是次一层的应用了。

​	需要说明的是，其实神经网络是有三种重要的应用的，分别是，图像处理，语言识别（或其他有上下文关系的识别），和人工智能。但是LeCun发表文章的时候，google的alpha-go还没有问世，再则人工智能因为牵扯到的细节与算法很多，已经脱离了普通开发者应用实践的程度，所以在这里才着重强调了前面两点。

```python
'''
In addition to beating records in image recognition1–4 and speech recognition5–7, it 
has beaten other machine-learning techniques at predicting the activity of potential
drug molecules8, analysing particle accelerator data9,10, reconstructing brain 
circuits11, and predicting the effects of mutations in non-coding DNA on gene 
expression and disease12,13. Perhaps more surprisingly, deep learning has produced 
extremely promising results for various tasks in natural language understanding14, 
particularly topic classification, sentiment analysis, question answering15 and 
language translation16,17.
（前言，第四段）

'''
```

#### 段落2： Supervised learning（监督学习）

​	机器学习从适用情景上分为三类，即监督学习（supervised），无监督学习（unsupervised）和强化学习（reinforcement），分别对应了训练中又标定的Y值（如最简单的线性回归，以及这篇文章应用到的绝大部分的实验），无标定的Y值需要方法自主分类（PCA，factor analysis等），以及Y值由训练的结果反馈而得，三种不同的情况。

​	文章的段落2，包含两大部分，第一部分从开始截止到第二页左上，介绍了监督学习中的一些基本概念与原理。第二部分从第二页右上开始（Many of the current practical ... ...）一直到整个段落结束，这一部分进一步比较了deep-learning和之前的‘shallow’ classifier 之间的差别。

​	关于基本原理部分，deep-learning在supervised的情境下，应用到的机器学习的基本原理与一般supervised learning是相同的（甚至选择了其中更为简单的样式），即线性回归的方法。没两个连续的单元之间，都是一层线性的关系，简单可以概括成Y=kX+b， k即权重，b是截距，然后每次计算之后比较预测的y与真实的y的差别（error），并通过随机（每次随机选择一个case）梯度（求偏导数的极端值找到可以最快变化的方向）速降（SGD）的方法，快速找到可以使误差最小的参数集。之后再用训练集得到的结果，比较一个同类而不在训练集中的样本集的表现（test），得到对于模型好坏的评估。

​	换言之，一个最简单的神经网络模型，或者说没有中间层的模型，就是简单的线性回归模型。而神经网络结构的复杂性和其独特的抽象能力，不是由于这些底层的优化器或者误差函数的改进实现的，而是通过网络这种想法实现的（参考下图，即原文的Figure1，c）。图中，神经元j的信号值，是由其下三个输入神经元以及他们对应的一组参数决定，所以最终J=w1·input(1) + wij·input(i) + w3*input(3) + b， 同理适用于J的附近兄弟3个神经元，以及j向上的H2层，一直到output层，此时output的结果已经是input经过三次抽象与优化得到得了，在比较这个output和最终监督值y的时候，output就可以捕获大量抽象的信息，而不简单只是各个输入的增加或者减少了。

​	![image] (tf_fmri_meg/reference/deep_learning/images/deep_learning_f1.png)

​	关于第二部分，比较deep-learning和其他线性的或者非线性和‘shallow classifier’之间的差别。文中举出了一个非常生动的萨摩耶-狼的例子，因为线性classifier会无差别的捕获不同x之间的差异，所以结果是，对于任何线性classifier，一直面向画面左侧的萨摩耶，和一只面向画面右侧的萨摩耶之间的差别，总是远大于他和一只面向左侧的狼的差别的。而即便是对于非线性的简单classifier（例如SVM），在实现区分抽象差异的功能的时候，依然会受限于有限的泛化能力，如利用高斯核函数Gaussian Kernal的非线性分类器只会把属于某类的Y的X在其对应向量面附近的点包含在自己的类当中（高斯指的是用正态分布的方式决定其他点与自己的相似度，距离远的点会被迅速排除出当前类中）。也因此，在使用的时候，对于应用者的经验（应用者能否准确的提取出样本与测试集中的有用的要素，应用者能否判断什么程度的泛化是合适的）就有了很高的要求。也就是在此处，作者再一次强调了deep-learning无需过多人为干预即可实现抽象提取的特性。

```python
'''
But this can all be avoided if good features can be learned automatically using a 
general-purpose learning procedure. This is the key advantage of deep learning.


... ...
Each module in the stack transforms its input to increase both the selectivity and the 
invariance of the representation. With multiple non-linear layers, say a depth of 5 to 
20, a system can implement extremely intricate functions of its inputs that are 
simultaneously sensitive to minute details — distinguishing Samoyeds from white wolves 
— and insensitive to large irrelevant variations such as the background, pose, lighting 
and surrounding objects.

（Supervised learning，结尾）

'''
```

​	总结来说，**deep-learning的每一层神经元之间，都是简单的线性回归关系，但是通过多层与复杂的链接，实现了抽象（非线性）提取的效果**。三者比较，deep-learning不仅在很大程度上保留了线性分离器，简单无需人为干涉实际计算过程的效果（对比非线性的核函数分离器），又实现了线性分离器所无法实现的抽象，保留有用的信息（如萨摩耶和狼的眼睛不同），压制无用的信息（狼/狗朝向）的非线性分离的效果。

#### 段落3： Backpropagation to train multilayer architectures（多层结构训练中的反向传播）

​	这一段落旨在说明，在网络结构中，如果在任意两个神经元之间都是使用简单的线性模型链接，那么在优化网落各部分的参数时，就可以利用到一个名为误差反向传播（back propagation of error）的性质。而这个性质也是保证网络结构可以被重复训练优化的基本。

​	![1546342275003](D:\Data\Works\TF_data_analyse\reference\%5CUsers%5C12440%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1546342275003.png)

​	上图（文中图1b）是一个最简单的所谓反向传播的例子，这个例子里面包含输入与输出只有3层神经元，每层一个，z相当于是预测值，z值要向数据中真实给出的每个样本的监督值靠拢，x值相当于输入值，即输入模型的每个变量x，虽然不知道z对于y的系数，但是z对于y的系数（z的偏导/y的偏导），可以由z关于x的系数表示出来。

​	不过，这里其实他使用的方法是，因为初始的时候，每一层都有一个默认值，所以他先用z和y的默认值，找到了当前最满足y能预测z的系数，又以y对x的默认值，找到当前最能让x预测y的系数，虽然有所谓反向传播作为一种理论依据，但**是实操起来却是期望优化其中每一小步，达到整体优化的效果**，这也是为什么他的所谓的反向传播的方法，曾经一度被学界否定的原因。因为任何企图通过局部优化的办法达到整体优化的尝试，在理论上都面临了很大的陷入局部最优解的风险。

​	

```python
'''
In particular, it was commonly thought that simple gradient descent would get trapped in poor local minima — weight configurations for which no small change would reduce the average error. In practice, poor local minima are rarely a problem with large networks.
'''
```

​	但是，实践最终战胜了所谓理论。这种简便易行的方法在实际操作中的优秀表现（特别是在抽象问题上的表现），为他们（Hinton，LeCun等）带来了转机。这也是Hinton一派的特点，作为deeplearning的泰斗人物，他们却少有真正的数学家或者计算学家，而是一只以工程师为主的队伍，善于解决现实问题，至于其中原理则往往会留给后人去发掘。在他们的文章中，也充满了如XX方法被认为并不好，但是我们发现他很快，所以我们就用了它之类的描述（例如ReLu等）。

#### 段落4：Convolutional neural networks（卷积神经网络）

​	卷积神经网络，是传统的深度学习网络的一种变体，通过模仿生物视觉加工（感受野）的方式，对于输入进行有规律的，按块提取，进而实现了大幅优化学习效率的结果。

​	如文中所说，之所以图片识别可以使用卷积网络进行优化，依赖于图片本身的四个主要特点，即：local connection，shared weight， pooling 和 the use of many leayers。 local connection指的是一般的图片识别任务，在2D空间上相邻的点，具有相似的特征（眼睛上的一个像素点它旁边的像素也很有可能是眼睛），shared weight 指的是，同一个特性，例如眼睛，出现在图片的不同位置，都是有可能的，也都应当被当做同一种特性来处理。Pooling，从概念上指的是复杂图片可以被抽象成简单要素的集合，而实际应用中，这种抽象转化为计算卷积（筛选特征）之后不同的卷积结果的最大值或者均值。最后，所以多layers则是作者用人类视觉系统做的一个类比，即图片的不同特征实在不同层抽象出来，初始可能只是简单的形状或者颜色，而高层的抽象则包含了更复杂的信息，例如形状的组合等。

```python
'''
The convolutional and pooling layers in ConvNets are directly inspired by the classic notions of simple cells and complex cells in visual neuroscience43, and the overall architecture is reminiscent of the LGN–V1–V2–V4–IT hierarchy in the visual cortex ventral pathway44.
'''
```

​	图2虽然给出了一个卷积的例子，但是在Hinton参加ImageNet的获奖的模型说明文章里面，对于感受野有更形象的描写，具体参看下图（ImageNet Classification with Deep Convolutional，Figure 3），这种中间暗两边亮的图形，或者监控不同色彩的图形，和对于感受野最初的研究发表的图是非常相似。

![1546346329137](D:\Data\Works\TF_data_analyse\reference\%5CUsers%5C12440%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1546346329137.png)



​	在应用层上，CNN模型用卷积层+池化层(convolution + poolling)代替了中间的训练层，通过利用图像本身的特点，简化训练过程（原始的模型是1个像素1个像素处理，而卷积之后可以用5x5,7x7的卷积核去处理，进而减少计算量并约束了重要的图形信息），一般的CNN网络包含一个输入层+数个卷积/池化层+1~2个全连接层（所谓全连接就是最初的神经网络结构，在出口处需要用全连接层与输出值对接）。

​	因为卷积的加入，在模型上卷积模型也会看上去更加复杂一些，下图是Hinton在上面文章中提到的自己的模型：

![1546531216389](D:\Data\Works\TF_data_analyse\reference\%5CUsers%5C12440%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1546531216389.png)

​	需要注意，卷积神经网络，只有在处理图像相关，或类图像问题上才能体现出其优越性来。且卷积神经网络不单单适用于2D图像，对于3D的图像，也有封装好的计算方法（这种3D的处理常常被用在视频解析，2D的图像+时间轴构成了3D，且相邻的时间，相似的图像点之间依然具有可被卷积的特性：local connection/shared weight等等。而在颜色有意义的2D图像识别中，也是需要3D的卷积核的，因为此时平面图本身就是3D的，每一个点都包含了一组RGB三个值，224x224的图像其实的数学展示是224x224x3），**我们这次要做MGE/fMRI的学习，也是要用到这个3D的卷积核**。如果仔细观察上面Hinton的模型，你会发现他也是在用3D卷积核。

​       此外，另一个特点是，因为是图像卷积处理，所以使用GPU（也就是独立显卡的计算核），可以大幅优化学习效率，在Hinton获奖的文章中，他用的两块GTX-580的GPU（8年前），实际上随着显卡制造业的飞速进展，现在大家已经在用1080TI或者更高级的特斯拉（专门用来机器学习的显卡）计算了。

#### 段落5：Image understanding with deep convolutional networks（深度学习网络在图像理解中的应用）

​	这一段并没有什么实际的内容，而是主要以炫耀作者自己的CNN被多少大型公司所采用为主。却是如作者所言，因为CNN在图片加工上带来的突破性进展，带来很很多实际的变革，也改变了很多大公司今后的计划。例如google利用CNN实现了自己的初代AI，alpha-go，只是这里有两点不同，第一alpha-go是强化学习，而不是作者所指的监督学习，其二google的CNN模型里面是只有卷积，没有池化的，原因是围棋盘上每一个子的信息都需要得到保存，而池化降维会导致重要信息的丢失。此外本文的作者Hinton后来也凭借自己在相关领域的杰出贡献，接任了google的人工智能研究组的组长工作。而另外一个代表是IBM公司，因为在各项业务上都没办法拔得头筹，所以IBM几乎变卖了自己的绝大部分业务，紧缩人员开始集中搞起了AI研究，并期待从中有所突破。至于显卡大厂英伟达（NVIDIA），近几年则为了抢占AI的硬件市场开发研究了大量专门用来机器学习的显卡（特斯拉系列，大概需要2W-7W一张普通的特斯拉卡），这种显卡具有强大的点阵计算能力，不过已经不再具有视频输出口，丧失了显卡最基本的现象功能了。

​	Deep-learning之所以会在这几年突然火爆起来，Hinton的CNN的贡献功不可没。不过很大原因也是因为随着智能设备（手机，电脑）和网络的普及，数据越来越多，获取数据也越来越方便。另一方面硬件设备的飞速发展也让之前理论上需要计算几年的模型可以在几分钟之内得到结果。两大优势，再加上CNN相对傻瓜（不需要对数据进行过多的预处理，也不需要应用者由强大的数学知识）的特点，使得CNN如雨后春笋在各行各业蓬勃发展起来。

​	在段落2中，他提到了两个用来防止过拟合的方法，一个是dropout，指的是训练的时候，每次随机关闭一部分神经元（如果部分神经元就可以表现的一样好，那就不需要全部的神经元了）。另一个是data augumentation，指的是在使用原始图片训练的同时，也是用原始图片剪裁后的图片训练（如果对少量信息也可以得到一样好的预测结果，就需要解读全部信息了）。这两种方法现在已经基本成了CNN的标配了。

#### 段落6：Distributed representations and language processing（分布表征与语言处理）

*注：后面的内容是Deep-learning的另一个分支，自然语言处理分支（RNN分支），RNN是用来处理连续体数据的，常见的如语音，语言等，我们的图像数据分析里面基本上不会用到这部分内容。不过这个也许会作为之后其他的分析的一些启蒙，特别其中的LSTM（long-short term memory）的概念，也是一定程度上借用了记忆相关的概念，所以还是给予文中内容尽量给出了充分的说明。但是另一方面，RNN比起经典网络（或者CNN）来说，结构要复杂的多，特别是LSTM，因而对于其中原理部分，文中的说明也比较简略，我们这里也就从简介绍了。

​	语言具有分布表征的特点，如何理解这种特点，以及如何表征语言，也就是各家机器学习争论的核心。

​	关于分布表征，作者在文中是这样解释的。

```python
'''
When trained to predict the next word in a news story, for example, the learned word vectors for Tuesday and Wednesday are very similar, as are the word vectors for Sweden and Norway. Such representations are called distributed representations because their elements (the features) are not mutually exclusive and their many configurations correspond to the variations seen in the observed data. These word vectors are composed of learned features that were not determined ahead of time by experts, but automatically discovered by the neural network. Vector representations of words learned from text are now very widely used in natural language applications.
'''
```

​	考虑一个切实的情况也许更容易理解一些，当出现一个英文句子填词，I want to go shopping on ___, 的时候，填入Wednesday（或者Tuesday/Holloween/Christmas）是合适的，填入apple/cat显然是不合适的。这说明Tuesday/Holloween/Christmas/Wednesday在某种意义上（虽然他们表面上看起来大相径庭）具有相似的表征（representation，这里表征的是一个确定的日期），这就是所谓的分布表征，不同的词，可以通过一定的表征方法，建立起他们之间的关联。

​	分布表征是所有语言学习共同的概念，deep-learning和传统方法的争论主要在于如何表征的问题上。

​	这里作者提到了一种叫做N-grams的方法，其实这也是之前最常用的自然语言处理的方法，这里简单介绍一下。假如有一个问题，判断以下句子的合法性，I wish to eat computer keyboard with Mary. 要如何判断呢，在2-grams的方法下，是这样进行处理的。句子的合理性 = 句子中所有词同时出现的合理性= 第一个词出现的概率x第一个词的基础上第二个词出现的概率x前两个词的基础上第三个词出现的概率....

​        写成公式是这样的

```python
s = 'I wish to eat computer keyboard with Mary'
P(s) = P(w1,w2,w3....w8) = p(w1) * p(w2|w1) * p(w3|w1,w2)...*p(w8|w1*w2*...*w7)
```

​       这是个完全标准的公式，不过代价是这个公式一旦展开计算就会变得十分复杂，所以2-grams相当于对于他做了一次简化，即认为只有相邻两个词（如果是3-gram就是相邻3个了，但是随着判断相邻的数量增加，计算量和训练量都会变得庞大起来）会互相影响，结果就把公式简化成了

```python
P(s) = P(w1,w2,w3....w8) = p(w1) * p(w2|w1) * p(w3|w2)...*p(w8|w7)
```

​       至于每个概率是多少，就需要训练的过程学习了，一般训练是通过通篇解析各种小说文本等，学习各种组合，校准每种组合出现的概率。针对我们的句子，一套学习成功的2-grams带入检测的结果，最后可能会发现这个句子虽然差不多每个配对都表现不错，但是其中的一对 p(computer|eat)概率太低了（computer几乎未曾出现在eat之后过，这在小说里也是常态），因为是乘法关系，所以显著拉低了整个句子的准确性，这样最终给出了不合法的评判。

​	2-grams是最常用的组合，当然也可以扩展到3-grams，4-grams或者更多，不过，就如文中提到的，对于一个总词数是V的词表，N-grams对应得组合数量是V^N，如果是泛用性的词库，例如搜狗的联想搜索，其中可能涉及到10000+的词，那10000^2 到10000^3的体量增加就不是所有用户能承受的了（用户想要脱机实现联想，就需要把这个概率表储存在本地，还需要一个足够大的内存加载，搜索这些词库）。不过搜狗现在其实是同时支持了云端联想和本地联想的，在联网的过程中打字，联想能力其实要更强一些。况且为了训练出绝大部分这种三连词的组合，也需要更多地训练集。而且，N-grams局限于N也无法解决距离比较远的线索问题。

​	RNN的出现就是为了解决N-gram等其他自然语言处理的经典机器解决不了的这些高级抽象问题的。需要说明，虽然在文中作者对于N-grams抱有一定程度上的贬低，不过（在落后的年代），N-grams是一个被认为普遍有效的方法，解决过很多如word的自动改错，自动联想等等。对于N-grams来说，表征可以理解成词对出现的概率，或者词对之间的距离（apple和orange距离eat近，Moon和April距离eat远）。而对于神经网络来说，这个表征则是高级神经元对于语义的抽象，且他被证实在处理一些复杂语言问题（例如翻译等）上有着更明显的优势。

#### 段落7：Recurrent neural networks（循环神经网络）

​	解决连续的数据，就需要RNN网络。因为RNN的基础网络有诸多问题，而LSTM（一种RNN网络的变体）又被发现具有广泛的适应性，所以现在谈到RNN网络，基本上都是在指LSTM了。

​	这里在继续介绍文章内容之前，先做一个什么情况该用什么网络的总结，因为到此为止我们已经介绍过全部deep-learning在的supervised环境下的应用了。

| 数据类型             | 例                                                           | 模型最优解                 | 对比经典机器学习                              |
| -------------------- | ------------------------------------------------------------ | -------------------------- | --------------------------------------------- |
| 非连续，结构化数据   | 用乘客性别，年龄，肤色，船舱号等预测生还率（泰坦尼克经典例） | DNN（传统deep-learning）   | 仅在维度多，数据量大情况下有优势，否则考虑SVM |
| 非连续，非结构化数据 | 2D图像识别动物（google-inception V3）                        | CNN（传统网络+卷积池化层） | 经典机器学习受到很大限制                      |
| 连续数据             | 自然语言翻译                                                 | LSTM（RNN的变体）          | 可实现经典机器学习无法实现的功能              |

​	希望上面的表格能够充分的说明不同的模型的应用环境。下面我们继续文中的内容。

​	RNN和经典网络或者CNN网络有着本质的不同，他并不是以从下到上逐层抽象，提取要素的方式实现的，而是以从时间的过去到未来，每走一步都记录和抽象一部分之前的信息实现的。文中的图5是一个最简单的RNN模型的样子，实际上大家已经不再使用这种基础RNN模型了，而是用一个更复杂的LSTM模型处理。不过LSTM的核心原理与这个RNN相同，所以这里先对这个基础RNN做一说明。

![1546529783844](D:\Data\Works\TF_data_analyse\reference\%5CUsers%5C12440%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1546529783844.png)

​	举个例子，比如，训练用的句子是“I want to eat pizza today.", 模型里面是三个参数（U,V,W）和一个记忆单元（中间那个小圆圈s，一般称为cell），第一进入模型的词(Xt-1)是I，此时的记忆单元是空的，经过(U,V)一通运算，得出了第一个Output（Ot-1）是like（当然这时候模型还没训练好呢，所以这个词基本上是随机的），然后第二个词输入了，是want，此时计算Ot-2的时候不仅会考虑 U对于输入词的运算，还要带上一部分来自上一步的计算（也就是f(W,St-1)）,最终预测Ot-2是sleep，如此直到句子读完，终止与句号，此时得到一套预测句子可能是"I like sleep for bed now"，再把这个句子和原始的输入句子作比较（然后发现基本全错了），之后再用之前所说的，反向传播的方式逐层优化，即想要让最后一个词在输入时pizza的时候出现today，需要什么样的(UVW)，如果要倒数第二个是pizza，又要对UVW进行怎样的调整，如此一致调整到第一次输入的时候，就算是一个训练结束了。

​	基本原理是上面的原理，当然实际操作的时候，会有一些更复杂的操作，例如图中只有一层细胞涉及到学习，但是这样能积累的知识是有限的，实际上会实际多层多个细胞，以充分提取不同细节。下图来自参考文献（Generating Text with Recurrent Neural Networks，图1）

​	![1546530720153](D:\Data\Works\TF_data_analyse\reference\%5CUsers%5C12440%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1546530720153.png) 

​	当然这还是一个未经过LSTM修饰的RNN模型，实际上经过LSTM之后，模型会变得更加复杂（具体有多复杂，可以看这篇文章Long short-term memory里面的插图...然后就会默默关掉文章感受岁月静好的）。不过LSTM却有效的通过在训练循环中加入遗忘机制，使得使用模型处理长句子时，不会因为经过的词太多模型太复杂而导致无法计算的情况。出去LSTM外，文中还介绍了一种名为Neural Turing Machine的训练模型，不过现在一说到自然语言处理，首选项依然是LSTM，好在LSTM也有封装好的工具可以调用，不用自己纠结其中复杂的遗忘机制，如此也算是一大便利了。对于LSTM的详情（我着实看不懂...），我们就不进一步介绍了。

#### 段落8：The future of deep learning（展望）

​	展望比起前言来，写的保守了很多。作者在这里再一次总结了两大突破进展，即图像识别（CNN），自然语言处理（RNN-LSTM），并且简单展望了一下在非监督学习和AI领域的应用。我们就不在作者本人都这么保守的基础上借题发挥了。不过这篇文章发表于2015年，那时deepLearning已经火遍了大型互联网公司，而在2016年4月，Alpha-go横空出世。先4:1大败李世石，之后又3:0柯洁，在公认最为复杂的围棋领域达到人类无法企及的地步。将深度学习推向了真正的热潮。随着硬件设备的发展，数据量的累积，以及无数致力于其中的研究者的努力，期望在接下来的10年里面deep-learning能把人类的认识推广到一个新高度吧。

### 总结

​	以下几条请作为重要结论保留，至于其中原理部分看个大概即可，因为本质上我们属于软件的应用方，不站在开发者的一端，最终我们做的工作是，用成熟的deep-learning在未知的领域（人类社会文化认知）找到了具有启发性的结果（期望），并且揭示了使用深度学习分析fMRI等神经图像所能带来的突破（期望）。至于如何使用成熟的deep-learning工具，就是反复的代码，调试，验证，继续代码的循环，在google和python的协助下，是一个不算简单但是可以完成的任务。

1. **深度学习（神经网络模型）最大的特点在于自动抽象**
2. **因此深度学习更适用于抽象表征的数据**
3. **通过大量的数据，与复杂的网络，深度学习可以在这些任务上表现出远超经典机器学习的水平**
4. **两个最成功的应用是，图像识别（卷积网络CNN），和自然语言处理（循环神经网路），我们要做的脑成像的处理，用的就是CNN网络**

### 其他

​	还有几件有趣的事，

​	第一，吴恩达现在在Coursara上又开了一门叫做deeplearning.ai的课程，和他的machine learning一样，属于所有同类里面讲的最清楚的一个人了。关于吴恩达其人，在中国属于最有名的几个研究机器学习的人物了，头衔也包括斯坦福的教授，googleAI的领导人（曾经，现在领导人是Hinton了），百度AI的领导人，Coursara的创始人等等。实际上，仅从科研贡献上，吴恩达是难以与文中提到的几位教授并肩的，无论是经典机器学习领域，还是新进的深度学习领域，他都没有发表过很具影响力的文章。在google的几年也没有突破进展，最终被Hinton替位。但是他的讲课能力确实很强，对于机器学习的推广起到了极为重要的作用（我自己的Machine Learning入门）也是通过他的课程，而且开创了惠及无数学子与希望提升自己技术技能的工薪族的Coursara，并且为人也比较正直（曾经亲自揭发了百度AI的一个参赛的造假项目），基于以上几点，还是十分值得尊敬的。这个课，连同他的经典机器学习课程，都推荐一下。

​	第二， 机器学习现在基本上都是给予python实现了，python属于非编译语言，语法简洁，对环境没有复杂的要求（一个3.6的编译器只要几M，对比JAVA编译之前要装JDK，运行时候还要JRE，加起来就1G还多）， 而且在github大家共同的努力下产生了成熟的应对各种复杂需求的包体（python上面甚至有专门处理fMRI原始数据的包体），并且可以很好的兼容其他语言（复杂的功能往往是在c++开发的基础上封装了一个python的调用外壳）或者工作环境（linux/windows下绝大部分python包是通用的）。可以预计在今后很长时间python也许都会是新功能的主流语言。我们之后的代码开也全部由python实现。说起来，这篇解析中的很多引用部分，虽然是原文引用，但是没有出现诡异的换行（如果你直接粘贴pdf就会有诡异的换行，毕竟他是两栏写的），或者不兼容字符的情况，也是用python写了一个小脚本实现的。

​	

```python
# MoonKuma
def format_string_para(file_name):
    long_str = ''
    with open(file_name,'r',encoding='utf-8') as file_op:
        for line in file_op.read():
            # line = line.strip()
            long_str = long_str+ line
    print(long_str.replace('\n',' '))
    return long_str

# test
format_string_para('test.txt')

#result should be like
'''
When trained to predict the next word in a news story, for example, the learned word vectors for Tuesday and Wednesday are very similar, as are the word vectors for Sweden and Norway. Such representations are called distributed representations because their elements (the features) are not mutually exclusive and their many configurations correspond to the variations seen in the observed data. These word vectors are composed of learned features that were not determined ahead of time by experts, but automatically discovered by the neural network. Vector representations of words learned from text are now very widely used in natural language applications.
'''
```

​	第三，deepLearning本质上是一套想法，不是一种计算机的编码方式，但是因为deepLearning的内部运算相对复杂，特别牵扯到多线程，CPU,GPU交互等内容，尽管其全部逻辑是开放的，但是在应用层上个体或者小公司几乎无法完成独立开发。而作为领跑全球DeepLearning的Google公司，在发售自己优质的服务的同时，也会不间断的开发的放出一部分封装好的用来计算deepLearning的程序包，我们常说的TensorFlow就是其中之一，也是为普罗大众接受和应用最多的一个。在tf的基础上，为了方便新手学习，google还封装了如Karas，Eager等进一步整合的包，实现了只用几行代码就完成一个简单的模型的训练的效果。封装的API越高级，使用越安全方便，但是代价是受到的束缚也就越多。我们此次的开发是在tf层实现的，最大程度利用google的TensorFlow，同时并没有使用进一步封装的API以保留一定的自由空间，如此以在中间调整模型，参数，并为不同的层造影等。