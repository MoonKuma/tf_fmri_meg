1.  Model, loss function and cost function for binary classification

   ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_basic\FireShot Capture 1 - Logistic Regression Cost Function - de_ - https___www.coursera.org_learn_neur.png)


model:  是一个线性方程套上sigmod方程的形式，保持了y预测值在0~1之间（含义是取y=1的概率）

loss：loss针对每一个样本，用(y'-y)^2的方式定义的loss会有局部最优的问题，所以该用了另外一种方式
$$
Loss(y',y) = -(ylogy' + (1-y)log(1-y'))
$$
cost function： 优化时最终需要保证整体训练集表现最优，所以通过定义cost（各个lost的某种综合体现）的方式定义，这里选用了各项的均值