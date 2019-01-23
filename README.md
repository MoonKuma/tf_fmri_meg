# tf_fmri_meg

文档结构

#### reference：参考文献

#### data_example: 数据样例

#### test_project: 测试工程

#### main_project：主工程

- ##### Project1_cnn_fmri

  - 说明：使用cnn模型，以fMRI-beta值作为输入，以condition（图片类型）作为输出，进行分类学习，比较分类训练结果与基础分析结果，各层权重
  - 数据结构
    - 样本数 = 被试数 * ran数 * condition数
    - 样本大小 = 64 * 64 * 33 （未经ROI的全脑beta值，注意不能加入normalize/smooth）
  - 模型
  - 结果

- ##### Project2_cnn_meg

  - 说明：使用cnn模型，以meg-sensor在各个时刻的信号值作为输入，以condition（图片类型）作为输出，进行分类学习，比较在各时刻分类器对于不同condition的区分能力
  - 数据结构
    - 样本数 = 被试数 *epoch数
    - 样本大小 = sensor数量
  - 模型
  - 结果

- ##### Project3_cnn_rsa

  - 说明：对fmri-beta数据，提取cnn各阶段抽象结果，对行为数据矩阵做rsa计算，同时比较meg各时刻结果
  - 数据结构：同p1,p2数据 
  - 模型
  - 结果



