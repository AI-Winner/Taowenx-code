'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-10-11 15:07:24
@LastEditTime: 2019-11-03 12:57:34
@LastEditors: Please set LastEditors
'''

#CSV数据集数据量太大，无法一次性纳入内存，通过pandas对象的iterator选项以及get_chunk实现分批次读入内存，最后将数据集concat起来
import pandas as pd
import time
path1 ='D:\TianchiMalwareDet\Data\security_train\security_train.csv'
path2 ='D:\TianchiMalwareDet\Data\security_test\security_test.csv'
print("path ok")
# iterator=True,得到一个迭代器,还有一个nrows指定读取的数目,还有一个chunksize每一次读多少.
# 文件预处理 CSV文件读取导入成列表形式

def FileChunker(path):
    temp=pd.read_csv(path,engine='python',iterator=True)
    loop=True
    chunkSize = 10000
    chunks = []
    while loop:
        try:
            chunk = temp.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    data = pd.concat(chunks, ignore_index= True,axis=0)
    print("chunker OK")
    return data


#把每个样本根据file_id进行分组，对每个分组把多个线程内部的API CALL调用序列排好，再把每个线程排好后的序列拼接成一个超长的字符串
def read_train_file(path):
    labels = []
    files = []
    data=FileChunker(path)
    #for data in data1:列表表头顺序为：file_id（文件编号）/label（病毒标签）/api（API名称，字符串）/tid（调用API的线程编号）/index（县城中API调用的顺序编号）
    goup_fileid = data.groupby('file_id')      #按文件编号file-id进行分组    每个分组下四个列 label（病毒标签）/api（API名称，字符串）/tid（调用API的线程编号）/index（县城中API调用的顺序编号） 
    for file_name, file_group in goup_fileid:   #每个id下按照（name和group进行迭代）
        #print(file_name)
        file_labels = file_group['label'].values[0]             # 获取label
        result = file_group.sort_values(['tid', 'index'], ascending=True)   #sort_value按某一列大小排序ascending = T为升序排序  先根据tid线程后index顺序排列
     #上一句排序结果为  id分组——按照tid排序——再按照index排序（嵌入的方式）  注意，result包含label
        api_sequence = ' '.join(result['api'])  #根据上面规则排序后的API序列
        labels.append(file_labels)  #labels文件只含标签项
        files.append(api_sequence)  #排列好的API序列
        #print(labels)   #回头删了
        #print(files)
    with open(path.split('/')[-1] + ".txt", 'a+') as f:
        for i in range(len(labels)):
            f.write(str(labels[i]) + ' ' + files[i] + '\n')
read_train_file(path1)   #读取训练数据
print("path1_txt OK")


#label和超长文本特征的转化为PKL文件，方便读入
#import pandas as pd
import numpy as np
import pickle
path1 ='D:\TianchiMalwareDet\Data\security_train\security_train.csv.txt'  #训练数据
def load_train2h5py(path):
    labels = []
    files = []
    with open(path) as f:
        for i in f.readlines():  #readlines读取所有行（直到结束符EOF）并返回列表。该列表可以由 Python 的 for... in ... 结构进行处理。如果碰到结束符 EOF 则返回空字符串。
            i = i.strip('\n')   #strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
            labels.append(i[0])  #label文件中训练集的第i[0]列
            files.append(i[1:])  #files文件保存除label之外的列  格式为列表形式
    labels = np.asarray(labels)   #将列表labels转换成数组格式
    with open("security_train.pkl", 'wb') as f:
        pickle.dump(labels, f)
        pickle.dump(files, f)
#load_train2h5py(path1)  #训练数据转换为pkl文件：security_train.pkl
#print("path1 PKL /train OK")
#load_train2h5py(path2)  #测试数据转换成pkl文件：security_test.pkl
#print("path2 PKL /test OK")


#测试文件做相同的处理
#把每个样本根据file_id进行分组，对每个分组把多个线程内部的API CALL调用序列排好，再把每个线程排好后的序列拼接成一个超长的字符串
def read_test_file(path):
#    labels = []
    files = []
    data=FileChunker(path)
    goup_fileid = data.groupby('file_id')                                                  # 不同的文件file_id
    for file_name, file_group in goup_fileid:
        result = file_group.sort_values(['tid', 'index'], ascending=True)   # 根据线程和顺序排列
        api_sequence = ' '.join(result['api'])
        files.append(api_sequence)
    with open(path.split('/')[-1] + ".txt", 'a+') as f:
        for i in range(len(goup_fileid)):
            f.write(' ' + files[i] + '\n')
#read_test_file(path2)   #读取测试数据
#print("path2_txt OK")

path22 ='D:\TianchiMalwareDet\Data\security_test\security_test.csv.txt'   #测试数据
#测试数据转换为pkl格式
def load_test2h5py(path):
    files = []
    with open(path) as f:
        for i in f.readlines():
            i = i.strip('\n')
            files.append(i[1:])
    #labels = np.asarray(labels)
    with open("security_test.pkl", 'wb') as f:
        pickle.dump(files, f)
load_test2h5py(path22)  #训练数据转换为pkl文件：security_test.pkl
print("path2 PKL /test OK")
