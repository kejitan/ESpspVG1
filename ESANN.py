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
import time

from  urllib import parse
from os.path import splitext, basename

from tqdm import tqdm
import six


in_dir = "./ANN/"

res = requests.get('http://localhost:9200')
print (res.content)
es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
client = Elasticsearch()

m = Mapping()
m.field('imgfile', 'text')
idx1 = Index('vgnum')
idx1.mapping(m)

def create_ESindex(out_fname) :
	file, ext = os.path.splitext(out_fname)
	CDict = {}
	CDict['imgfile'] = file + ".jpg"
	f = open(in_dir + out_fname, "r")
	annlines = f.readlines()
	classnum_list = []
	classname_list = []
	i = 0
	for line in annlines[1:] : # First line is labels
		if i > 10 : 
			break
		fields = line.split(',')
		classnum_list.append(fields[0])

	CDict["classnum"] = classnum_list
	obj_str = json.dumps(CDict)
	es.index(index='vgnum', body=json.loads(obj_str))

inp = glob.glob(os.path.join(in_dir, "*.txt")) 
count = 0
for i, inp in enumerate(tqdm(inp)):
	if isinstance(inp, six.string_types):
		out_fname = os.path.basename(inp)
		create_ESindex(out_fname)



    
