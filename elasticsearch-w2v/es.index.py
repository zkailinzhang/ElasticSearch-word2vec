
from elasticsearch import Elasticsearch
import jieba 
from gensim import models 
import json
from gensim.models import doc2vec
from elasticsearch.helpers import bulk
import numpy as np 
import pandas as pd 

ES = Elasticsearch([{'host':'127.0.0.1','port':9200}]) 
indexName = 'oto1'

import pickle 

item_tags_map_A = pickle.load(open("item_tags_map_A.pkl",'rb'))
tags_map_B = pickle.load(open("tags_map_B.pkl",'rb'))
tags_sum_item_vec = np.load('./tags_sum_vecs.npy')

def create_index():

    setting = {
        "settings": {
            "number_of_replicas": 0,
            "number_of_shards": 2
        },
        "mappings": {
           
                "properties": {
                "idname": {
                    "type": "text"  #text string keyword
                },
                "idx": {
                    "type": "text"
                },
                "feature": {
                    "type": "text"
                },
                "feature_vector": {
                    "type": "dense_vector",
                    "dims": 200
                }
           
            }
            
        }
    }
    res = ES.indices.create(index=indexName, body=setting)





def bulk_index_data():
    """
    将数据索引到es中，且其中包含描述的特征向量字段
    """

    requests = []
    for iid,tag in item_tags_map_A.items():
        if tag =='NaN':continue
        
        test_tags_index =  tags_map_B[item_tags_map_A[iid] ]
        test_tag_vec = tags_sum_item_vec[test_tags_index]

        request = {'_op_type': 'index',  # 操作 index update create delete  
                   '_index': indexName,  
                   '_source':
                       {
                           'idname': iid,
                           'idx': test_tags_index,
                           'feature': tag,
                           'feature_vector': test_tag_vec,
                       }
                   }
        requests.append(request)
        
        if len(requests)>10:break

    success, _ = bulk(ES, requests)



def test():
  
    es = ES
    
    test_item_id = '33635262-PG_661feeb143592e50113545382f5977b6'
    test_tags_index = tags_map_B[item_tags_map_A[test_item_id] ]
    test_tag_vec = tags_sum_item_vec[test_tags_index]

    try:
            
            
        resp = es.search(index=indexName, body={
            "_source": ["idname", "feature"],
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "1/(1+l2norm(params.queryVector, doc['feature_vector']))",
                        
                        "params": {
                            "queryVector": test_tag_vec
                        }
                    }
                }
            }
        })
        print("可能获得的是：", end=" ")
        for hit in resp["hits"]["hits"]:
            print(hit["_source"]["idname"],hit["_source"]["feature"], end="\t")
        print("\n")
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    
    #create_index()
    #bulk_index_data()
    test()