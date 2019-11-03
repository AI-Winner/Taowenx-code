'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-10-11 15:15:35
@LastEditTime: 2019-10-29 20:14:35
@LastEditors: Please set LastEditors
'''
#Use the TfidfTransformer in sklearn to count the tf-idf weights of 
#each word or contiguous word (n-gram) in the text to assess how important a word is 
# to a document set or one of the documents in a corpus:

#训练数据处理
#import DataImport  #关联数据导入文件
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import numpy  as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn import manifold
start = time.clock()
with open("security_train.pkl", "rb") as f:   # 加载PKL文件，训练数据
    labels = pickle.load(f)
    files = pickle.load(f)
print("start tfidf...")
vectorizer = TfidfVectorizer(ngram_range=(1, 4), min_df=0.0, max_df=1.0,)  # tf-idf特征抽取ngram_range=(1,5)
train_features = vectorizer.fit_transform(files)          # 将api长序列进行TFIDF，作为样本特征  每行api组合为一个长文本序列  抽取长文本的特征 生成特征向量 
#以上每行”label+长文本（api组合）“转换为了“label+特征向量（每行特征向量数目不同，原因是因为所包含API名称的数目不同 所以抽取出来的特征数也不一样）”
print(" TFIDF Finish!")
# 降维
#执行映射，我们把维度降为500
print("demension reduction start!")
print("phase 1")
transformer = GaussianRandomProjection(n_components=500)
train_features = transformer.fit_transform(train_features)
#transformer = SparseRandomProjection(n_components=500, random_state=0)
#train_features = transformer.fit_transform(train_features )
#transformer=PCA(n_components=500)
#train_features=transformer.fit(train_features)
print("phase 2")
print("demension reduction start!")   
print(train_features.shape)                                       #   打印出转换后的向量形态
with open("tfidf_feature_train1029.pkl", 'wb') as f:  # pickle
    pickle.dump(train_features, f)     #序列化对象  将对象保存到文件中
with open("tfidf_feature_train1029.pkl", 'rb') as f:
    train_features = pickle.load(f)    #反序列化  
#（为什么要做序列化和反序列化操作：为了保存）
#打印后的被存成了numpy二维数组形式，这里相当于做了一个词嵌入
print(train_features.shape)
print(" PKL Finish!")
end = time.clock()
print("time is:",int(end-start))


#测试数据处理
#start = time.clock()
#with open("security_test.pkl", "rb") as f:   # 加载PKL文件，训练数据
#    labels = pickle.load(f)
#    files = pickle.load(f)
#print("start tfidf...")
#vectorizer = TfidfVectorizer(ngram_range=(1, 4), min_df=0.0, max_df=1.0, )  # tf-idf特征抽取ngram_range=(1,5)，取全集
#test_features = vectorizer.fit_transform(files)          # 将api长序列进行TFIDF，作为样本特征
#print(" TFIDF Finish!")
# 降维
#执行映射，我们把维度降为500
#print("demension reduction start!")
#print("phase 1")
#transformer_test = GaussianRandomProjection(n_components=500)
#test_features = transformer_test.fit_transform(test_features)
#transformer = SparseRandomProjection(n_components=500, random_state=0)
#train_features = transformer.fit_transform(train_features )
#transformer=PCA(n_components=500)
#train_features=transformer.fit(train_features)
'''
print("phase 2")
print("demension reduction start!")   
print(test_features.shape)                                       #   打印出转换后的向量形态
with open("tfidf_feature_test.pkl", 'wb') as f:  # pickle
    pickle.dump(test_features, f)     #序列化对象  将对象保存到文件中
with open("tfidf_feature_test.pkl", 'rb') as f:
    test_features = pickle.load(f)    #反序列化  
#（为什么要做序列化和反序列化操作）
print(test_features.shape)
print(" PKL Finish!")
end = time.clock()
print("time is:",int(end-start))
'''