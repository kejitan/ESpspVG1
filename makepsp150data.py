
import pandas as pd
import re
import json
from pandas.io.json import json_normalize

import cv2

json.load((open('./image_data.json')))
with open('./image_data.json') as json_string:
    image_str = json.load(json_string)

image_str_data = json_normalize(image_str)
num_images = len(image_str_data)
print("Num VG image = {}".format(num_images))

for i in range( 0, num_images) :
    url = image_str_data.at[i, 'url']

    subfields = url.split('/')
    subdir = subfields[-2]
    if subdir == 'VG_100K' :
        imagedir = 'images'
    elif subdir == 'VG_100K_2' :
        imagedir = 'images2'
    image_name = subfields[-1]
    #print("image_name = {}", image_name)
    image_path = './'
    image_path = image_path+imagedir+'/'+subdir+'/'+image_name
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(img, (473,473), interpolation = cv2.INTER_AREA)
    cv2.imwrite('./assets/'+image_name, resized) 




