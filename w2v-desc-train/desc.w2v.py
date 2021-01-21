# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Droid Sans Fallback']
matplotlib.rcParams['axes.unicode_minus']=False

# plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
# plt.rcParams['axes.unicode_minus']=False


import pandas as pd
import numpy as np

import textrank4zh

file = pd.read_csv('/home/tv/Downloads/query_result_desc.csv',)
desc = file['description']

desc_list = list(desc)



desc_ll = list(set(desc_list))
#des4c_ll.remove('nan')
for v in desc_ll:
    if 'nan' in str(v):
        desc_ll.remove(v)

import re


#去除中文标点符号
from zhon.hanzi import punctuation

desc_lt =list( map(lambda x:
    re.sub("[%s]+"%punctuation," ",x), desc_ll))



import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs,sys


desc_lt_cut = list(map( lambda x: ' '.join(jieba.cut(str(x),cut_all=False)), desc_lt))

with open('./item_only_desc_words.txt','w') as ff:
    ff.write(str(desc_lt_cut))




import multiprocessing
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

#sentence = LineSentence('./item_only_desc_words.txt')
sentence = [x.split() for x in desc_lt_cut ]
#读取停用词
stop_words = [] 
with open('/home/tv/code/word2vec/word2vec-tensorflow-cn/stop_words.txt',"r",encoding="UTF-8") as f:
    line = f.readline()
    while line:
        stop_words.append(line[:-1])
        line = f.readline()
stop_words = set(stop_words)
print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))

sentence_ll = []
for x in desc_lt_cut:
    sentence_ll.append( list(filter(lambda y: not y in stop_words, x.split())))


model = Word2Vec(sentence_ll,size=32,window=5,min_count=1,workers = multiprocessing.cpu_count())


model.save('./item_only_desc.text.model')
model.wv.save_word2vec_format('./item_only_desc.text.vector',binary=False)



from sklearn.decomposition import PCA
from matplotlib import pyplot

xvec = model[model.wv.vocab]



pca= PCA (n_components=2)
rst = pca.fit_transform(xvec)

pyplot.scatter(rst[:,0],rst[:,1])
words = list(model.wv.vocab)

for i ,word in enumerate(words):
    pyplot.annotate(word,xy=(rst[i,0],rst[i,1]))
pyplot.show()
#pyplot.savefig()

import gensim

mode = gensim.models.Word2Vec.load('./item_only_desc.text.model')
print(mode.similarity('电影','体育'),
mode.similarity('电影','电视剧'),
mode.similarity('动画','儿童'),
)

word = model.most_similar(u"足球")
for t in word:
    print (t[0],t[1])
