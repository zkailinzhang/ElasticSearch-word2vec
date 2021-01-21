from sklearn.decomposition import PCA
from matplotlib import pyplot
import gensim 
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Droid Sans Fallback']
matplotlib.rcParams['axes.unicode_minus']=False
model = gensim.models.Word2Vec.load('./item_only_desc.text.model')
# xvec = model[model.wv.vocab]



# pca= PCA (n_components=2)
# rst = pca.fit_transform(xvec)

# #pyplot.scatter(rst[:,0],rst[:,1])
# words = list(model.wv.vocab)

# for i ,word in enumerate(words[:1000]):
#     pyplot.annotate(word,xy=(rst[i,0],rst[i,1]))
# pyplot.show()


from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import random
 
#因为词向量文件比较大，全部可视化就什么都看不见了，所以随机抽取一些词可视化
words = list(model.wv.vocab)
random.shuffle(words)
 
vector = model[words]
tsne = TSNE(n_components=3,init='pca',verbose=1)
embedd = tsne.fit_transform(vector)
 
#可视化
plt.figure(figsize=(14,10))
plt.scatter(embedd[:300,0], embedd[:300,1],embedd[:300,2])
 
for i in range(300):
    x = embedd[i][0]
    y = embedd[i][1]
    z = embedd[i][2]
    plt.text(x, y, words[i])
    plt.text()

plt.show()