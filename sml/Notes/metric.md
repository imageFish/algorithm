## Metrics
$$
acc = \cfrac{TP+TN}{TP+TN+FP+FN}\\
percision = \cfrac{TP}{TP+FP}\\
recall = \cfrac{TP}{TP+FN}\\
f1 = \cfrac{2}{\cfrac{1}{p}+\cfrac{1}{r}}=\frac{2pr}{p+r}\\
f_\beta = (1+\beta^2)\cfrac{pr}{\beta^2p+r}\\
Micro_{F1} = \frac{2\sum p_i * \sum r_i}{\sum p_i + \sum r_i}\\
Macro_{F1} = \text{mean}{F1_i}
$$
## ROC curve (Receiver operating characteristic curve)
- x: False Positive Rate=$\frac{FP}{FP+TN}$
- y: True Positive Rate=$\frac{TP}{TP+FN}$
- TPRate的意义是所有真实类别为1的样本中，预测类别为1的比例。
- FPRate的意义是所有真实类别为0的样本中，预测类别为1的比例。

AUC的物理意义为，任取一对（正、负）样本，正样本的score大于负样本的score的概率。ROC的积分

AUC计算，复杂度O(mn)
$$
AUC = \frac{\sum PositiveRank - M(1+M)/2}{M*N}
$$
注：排序好之后正样本i的对数，$rank-1-(M_i-1)$其中$M_i$为正样本i在所有正样本中的排序，FPR与TPR好像不能直接比较，但是实际上已经把他当成了在预测中正确和错误的比例

## tf-idf
$$
\text{TF-IDF}_w = tf * idf\\
idf_w = \log_{10} \frac{|D|}{|D_w|}\\
tf_w = \frac{\text{NUM}_w}{\text{NUM}_{AllWord}}
$$
idf为所有文档除以包含该单词的文档对10的对数，tf是该单词的词频