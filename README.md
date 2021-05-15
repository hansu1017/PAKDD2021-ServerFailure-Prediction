# PAKDD2021-FailureServer-Prediction
PAKDD2021第二届阿里云智能运维算法大赛第23名解决方案

赛题与数据下载链接：https://tianchi.aliyun.com/competition/entrance/531874/information

评分函数借鉴：https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.3.166144c3LEoTaC&postId=200049

方案和经验总结blog：https://zhuanlan.zhihu.com/p/370924882

# 解决方案

address_feature.ipynb为address数据集的特征构造，由于运行时间较长，将address特征单独运行并保存为.csv文件以便进行其他分析；set_model.ipynb为kernel和mce数据集的特征构建以及模型训练过程；predict.ipynb为模拟线上数据流评测。本次比赛中构建特征和建模方法如下：

## 1.特征工程：

（1）对于kernel数据集，将数据按照serial_number、manufacturer、vendor和collect_time分组求和，并统计2分钟内的日志数，之后对每个server的2min日志数统计均值、中位数和加和作为新特征，同时对kernel表中原有的24个特征进行筛选，删除了方差为0以及相关系数大于0.8且再建模过程中重要性极低的特征。

（2）对于address数据集，分别统计每个server的每2分钟内row和col的数量、重复出现（出现次数大于1）的数量以及最大重复次数作为特征。

（3）对于mce数据集，将transaction和mca_id两个分类变量变成多个哑变量，以2分钟为间隔计算各哑变量相应的数量，并在多次建模后删除个别重要性极低的变量。

## 2.建模

本次使用catboost进行建模，将manufacturer和vendor作为两个分类变量处理（cat_features），最终共32个特征。用初赛1-5月数据以及初赛b榜7月数据前20天作为训练集，初赛b榜7月份最后十天的数据作为验证集，模型分为分类和回归两部分：第一步进行分类预测，将距离故障时间七天以内的日志的标签设为1，其他日志标签为0。由于正负样本量极不平衡，以正负样本1：10的比例在负样本中抽样，并与所有正样本合并作为训练样本，最终得到预测标签为1的所有测试集样本，即预测未来7天会故障的服务器。第二步训练集变为第一步样本中的所有正样，因变量为距离故障的时间（单位为分钟），用catboost回归来确定故障服务器距离故障的具体时间（pti）。
 
 # Python库依赖环境
     numpy == 1.18.5
     pandas == 1.0.5
     catboost == 0.25.1

# 声明
感谢BruceQD老师对本次比赛的指导，本项目代码仅供学习参考，如对方案或代码有任何疑问请邮件联系：hansu1017@163.com
