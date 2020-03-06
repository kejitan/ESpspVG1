#!/usr/bin/env python

from PIL import Image 
import glob, os
from tqdm import tqdm
import six
#import cv2
import pandas as pd
from keras_segmentation.data_utils.data_loader import get_image_array, get_segmentation_array
import numpy as np

import re
import json
from pandas.io.json import json_normalize
import time
import multiprocessing

def seg2ann(seg_file) :
	data = pd.read_csv('./PSPindexClass.csv')
	cols = ['Idx','Ratio','Train','Val','Stuff','Name']
	CNames = np.empty(150, dtype=np.object)
	for k in range(150):
		CNames[k] = data['Name'].iloc[k]

	seg_labels = get_segmentation_array(seg_file, 150, 473, 473, no_reshape=True)

	CN = np.empty(150,dtype=np.object)
	for i in range(CN.shape[0]):
		CN[i] = []
	xsumavg = np.zeros(150)
	ysumavg = np.zeros(150)
	xsum = 0
	ysum = 0

	for k in range (150):
		CN[k].append(k+1)  # class num
		CN[k].append(0)    # classs val CN[1]
		CN[k][1] = np.sum(seg_labels[:,:,k], axis=(0,1))
		if CN[k][1] > 0 :
			for i in range(473):
				for j in range(473):
				    if (seg_labels[i, j, k]) == 1 :
				        xsumavg[k] = xsumavg[k] + j 
				        ysumavg[k] = ysumavg[k] + i
				        
			xsumavg[k] = xsumavg[k]/CN[k][1] 
			ysumavg[k] = ysumavg[k]/CN[k][1]

	CDict = {}
	for k in range(150):
		if CN[k][1] != 0:
			centroidx = xsumavg[k]
			centroidy = ysumavg[k]
			CDict[CN[k][1]] = [ (CN[k][0]), (CN[k][1].astype(int)), CNames[k] ]

	return CDict




