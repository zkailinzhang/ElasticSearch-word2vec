# -*- coding:utf-8 -*-


import pandas  as pd 
import numpy as np
import pickle 
import json
import pandas_profiling as pp
import faiss
import re



def compute_ngrams(word, min_n, max_n):
    #BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
    extended_word =  word
    ngrams = []
    for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
        for i in range(0, len(extended_word) - ngram_length + 1):
            ngrams.append(extended_word[i:i + ngram_length])
    return list(set(ngrams))


def wordVec(word, wv_from_text, min_n = 1, max_n = 3):
    
   
    word_size = wv_from_text.wv.syn0[0].shape[0]   
   
    ngrams = compute_ngrams(word,min_n = min_n, max_n = max_n)
    
    if word in wv_from_text.wv.vocab.keys():
        return wv_from_text[word]
    else:  
        
        word_vec = np.zeros(word_size, dtype=np.float32)
        ngrams_found = 0
        ngrams_single = [ng for ng in ngrams if len(ng) == 1]
        ngrams_more = [ng for ng in ngrams if len(ng) > 1]
      
        for ngram in ngrams_more:
            if ngram in wv_from_text.wv.vocab.keys():
                word_vec += wv_from_text[ngram]
                ngrams_found += 1

        if ngrams_found == 0:
            for ngram in ngrams_single:
                word_vec += wv_from_text[ngram]
                ngrams_found += 1
        if word_vec.any():
            return word_vec / max(1, ngrams_found)
        else:
            raise KeyError('all ngrams for word %s absent from model' % word)



def compute_tags_sum_vec():

    item = pd.read_csv("query_items_test.csv")

    item["item_id"]= item["item_id"].map(str).str.cat(item['item_type'],sep='-')
   
    item['tags'].fillna('NaN',inplace=True)  
    item['tags'].fillna('NaN',inplace=True)
 
    item_tags_map_A = {val["item_id"]:val["tags"] for ii,val in item[['tags','item_id']].iterrows()}

    item_tags_map_A_inv={}
    for k,v in item_tags_map_A.items():
        vals =[]
        if v in item_tags_map_A_inv.keys():       
            vals = item_tags_map_A_inv[v]
           
        vals.append(k)
        item_tags_map_A_inv[v]  = vals


    tags_map_B = {val:ii for ii,val in enumerate(set(item['tags'].dropna()))}
    tags_map_B['PAD'] = len(tags_map_B)+1

    from itertools import izip
    tags_map_B_inv = dict(izip(tags_map_B.itervalues(), tags_map_B.iterkeys()))


    from gensim.models import KeyedVectors

    file = 'Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt'
    wv_from_text = KeyedVectors.load_word2vec_format(file, binary=False) 
    wv_from_text.init_sims(replace=True)  


    tags_nan_vec=np.zeros([200], dtype=np.float32)
    

    tags_sum_item_vec = np.zeros([len(tags_map_B_inv)+1,200], dtype=np.float32)

    for k,v in tags_map_B_inv.items(): 
        
        if v!="PAD": 
           
            v= re.sub(u"([\u0030-\u0039\u0041-\u005a\u0061-\u007a\uAC00-\uD7A3]|[\" +\"])","",v) 
            v= re.sub(r'\|\|','|',v) 
            if (v ==['']) or (v=='') or (v==' ')or (v==[' ']) or (v=='|'):
                tags_sum_item_vec[k] = tags_nan_vec 
                continue     
            v= v[:-1] if v[-1]=='|'  else v 
            v= v[1:] if v[0]=='|' else v 
            # '算法||'  '算法|'  '色方式||撒'

            tags_sum = [wordVec(x, wv_from_text, min_n = 1, max_n = 3)  for x in   v.split('|')] 
            tags_sum_item_vec[k] = np.mean (tags_sum,axis=0)
        else: 
            tags_sum_item_vec[k] = tags_nan_vec

    filename = './tags_sum_vecs.txt'
    with  open(filename, 'w', encoding='utf-8') as ff:
        ff.write(tags_sum_item_vec)
    np.save('./tags_sum_vecs.npy',tags_sum_item_vec)

    output = open("item_tags_map_A.pkl",'wb')
    pickle.dump(item_tags_map_A,output)
    output = open("tags_map_B.pkl",'wb')
    pickle.dump(tags_map_B,output)
    output = open("tags_map_B_inv.pkl",'wb')
    pickle.dump(tags_map_B_inv,output)
    output = open("item_tags_map_A_inv.pkl",'wb')
    pickle.dump(item_tags_map_A_inv,output)



def test(test_item_id):

    item_tags_map_A = pickle.load(open("item_tags_map_A.pkl",'rb'))
    item_tags_map_A_inv = pickle.load(open("item_tags_map_A_inv.pkl",'rb'))
    tags_map_B = pickle.load(open("tags_map_B.pkl",'rb'))
    tags_map_B_inv = pickle.load(open("tags_map_B_inv.pkl",'rb'))

    tags_sum_item_vec = np.load('./tags_sum_vecs.npy')

    index = faiss.IndexFlatL2(200)
    index.add(tags_sum_item_vec)


    print([item_tags_map_A[test_item_id]])
    test_tags_index = tags_map_B['PAD'] if pd.isnull([item_tags_map_A[test_item_id]] ) else tags_map_B[item_tags_map_A[test_item_id] ]

    test_tag_vec = tags_sum_item_vec[test_tags_index]

    D,I = index.search(np.reshape(test_tag_vec,[1,200]),10)

    print('测试item_id {},tags:{}'.format(test_item_id,item_tags_map_A[test_item_id]))
    print(I[-5:])

    for i in  range (len(I[0])):
        r_vec = tags_sum_item_vec[I[0][i]]
        r_tags = tags_map_B_inv[I[0][i]]
        r_item_list= item_tags_map_A_inv.get(r_tags)
        print('距离第 {},距离 {}, item_id {},tags:{}'.format(I[0][i],D[0][i],r_item_list,r_tags))



if __name__ == "__main__":
    #step1
    #compute_tags_sum_vec ()
    #step2
    test_item_id = '33635262-PG_661feeb143592e50113545382f5977b6'
    #'国内|湖北|医疗'
    test_item_id = '33635568-PG_661feeb143592e50113545382f5977b6'
    #国内|体育|赛事
    test_item_id = '33637198-PG_661feeb143592e50113545382f5977b6'
    #手游|普通游戏|和平精英
    #test_item_id = '33646518-PG_661feeb143592e50113545382f5977b6'
    #为空
    test(test_item_id)

