import os
import gc
import json
import requests
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import bulk
from elasticsearch_dsl.query import Bool, MultiMatch
from elasticsearch_dsl.search import Search, MultiSearch
from elasticsearch_dsl import Mapping, Keyword, Nested, Text
from elasticsearch_dsl import Index, analyzer, tokenizer
from elasticsearch_dsl import Q
import glob

res = requests.get('http://localhost:9200')
print (res.content)
es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
client = Elasticsearch()

id = Index('vgnum') #  
id.delete(using=client)

#id = Index('idxi20') #  
#id.delete(using=client)

#id = Index('idxo20') # 
#id.delete(using=client)

