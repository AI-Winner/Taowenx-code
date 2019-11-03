'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-10-11 15:18:43
@LastEditTime: 2019-10-29 20:26:36
@LastEditors: Please set LastEditors
'''

#XGBoost achieves multiple classifications
#import DataProcess
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import csv
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import numpy  as np
from sklearn.externals import joblib
start = time.clock()
with open("security_train.pkl", "rb") as f:   # 加载PKL文件，训练数据
    labels = pickle.load(f)
with open("tfidf_feature_train1029.pkl", 'rb') as f:  # 加载
    train_features = pickle.load(f)
print(train_features.shape)
print(labels.shape)
print("File Loading Finish")
# 5-Fold Cross
skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
print("StratifiedKFold Finish ")
#按enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。 
for i, (tr_ind, te_ind) in enumerate(skf.split(train_features, labels)):           # 迭代训练
    X_train, X_train_label = train_features[tr_ind], labels[tr_ind]
    X_val, X_val_label = train_features[te_ind], labels[te_ind]
    print('FOLD: {}'.format(str(i)))                                                            # 训练第几次
    print( len(tr_ind),len(te_ind))                                                                
    # XGB
    dtrain = xgb.DMatrix(X_train, label=X_train_label)                              # XGBoost自定义了一个数据矩阵类DMatrix，优化了存储和运算速度
    dtest = xgb.DMatrix(X_val, label=X_val_label)
    # dout = xgb.DMatrix(out_features)
    ## 参数设置
    #max_depth： 树的最大深度。缺省值为6，取值范围为：[1,∞]
    #eta：为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。 
    #eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
    #silent：取0时表示打印出运行时信息，取1时表示以缄默方式运行，不打印运行时信息。缺省值为0
    #objective： 定义学习任务及相应的学习目标，“binary:logistic” 表示二分类的逻辑回归问题，输出为概率。    
    param = {'max_depth': 5, 'eta': 0.1, 'eval_metric': 'mlogloss', 'silent': 1, 'objective': 'multi:softprob','num_class': 8, 'subsample': 0.8,'colsample_bytree': 0.85}  
    evallist = [(dtrain, 'train'), (dtest, 'val')]  # 测试 , (dtrain, 'train')
    num_round = 300  # 循环次数
    print('XGB begin')
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50)
    print("XGB finish")
    # dtr = xgb.DMatrix(train_features)？？？？
    pred_val = bst.predict(dtest)               # 验证集预测错误概率
   #print ('predicting, classification error=%f' % (sum( int(pred_val[i]) != int(X_val_label[i]) for i in range(len(X_val_label)))  /len(X_val_label)))
print("start saving model")
joblib.dump(bst, "train_model1029.m")
print("done！")
end = time.clock()
print("time is:",int(end- start))





