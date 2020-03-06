#!/usr/bin/env python

from PIL import Image 
import glob, os
from tqdm import tqdm
import six
import pandas as pd
from data_utils.data_loader import get_image_array, get_segmentation_array
import numpy as np

import re
import json
from pandas.io.json import json_normalize
import time
import multiprocessing

def seg2ann(seg_dir="../images/segmentationpng/", ann_dir="../images/annotation/") :
	data = pd.read_csv('../PSPindexClass.csv')
	cols = ['Idx','Ratio','Train','Val','Stuff','Name']
	CNames = np.empty(150, dtype=np.object)
	for k in range(150):
		CNames[k] = data['Name'].iloc[k]

	seg_file = glob.glob(os.path.join(seg_dir, "*.png")) 
	for i, seg_file in enumerate(tqdm(seg_file)):
		if isinstance(seg_file, six.string_types):
			out_fname = os.path.basename(seg_file)
			file, ext = os.path.splitext(out_fname)
			fout = open(ann_dir + file + ".txt", "w")

			seg_labels = get_segmentation_array(seg_file, 150, 473, 473, no_reshape=True)

			CN = np.empty(150,dtype=np.object)
			for i in range(CN.shape[0]):
				CN[i] = []
			xsumavg = np.zeros(150)
			#print("xsumavg = {0}".format(xsumavg))
			ysumavg = np.zeros(150)
			xmin = np.zeros(150)
			ymin = np.zeros(150)
			xmin.fill(473)
			ymin.fill(473)
			xmax = np.zeros(150)
			ymax = np.zeros(150)
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
						        if i < ymin[k]:
						            ymin[k] = i
						        if i > ymax[k]:
						            ymax[k] = i
						        if j < xmin[k]:
						            xmin[k] = j
						        if j > xmax[k]:
						            xmax[k] = j
						        
					xsumavg[k] = xsumavg[k]/CN[k][1] 
					ysumavg[k] = ysumavg[k]/CN[k][1]

			CDict = {}
			for k in range(150):
				if CN[k][1] != 0:
					boxarea = (xmax[k]-xmin[k])*(ymax[k]-ymin[k])
					if boxarea > 0:
						density = CN[k][1]/boxarea
					centroidx = xsumavg[k]
					centroidy = ysumavg[k]
					CDict[CN[k][1]] = [ "{:3d}".format(CN[k][0]), "{:08d}".format(CN[k][1].astype(int)), "{:04d}".format(xmin[k].astype(int)), "{:04d}".format(xmax[k].astype(int)), "{:04d}".format(ymin[k].astype(int))
, "{:04d}".format(ymax[k].astype(int)), "{:07d}".format(boxarea.astype(int)) 
, "{:.4f}".format(density.astype(float)), "{:04d}".format(centroidx.astype(int)), "{:04d}".format(centroidy.astype(int)) 
, CNames[k] ]
			fout.write("classnum, classval, xmin, xmax, ymin, ymax, boxarea, density, centroidx, centroidy, classname\n")
			for key in sorted(CDict.keys(), reverse=True) :
				if key != 0 :
					listToStr = ','.join(map(str, CDict[key]))
					fout.write(listToStr+'\n') 
			fout.close()

if __name__ == "__main__": 
	processes = [] 

	#t1 = multiprocessing.Process(target=seg2ann, kwargs={'seg_dir': "../images/segTrainpng1", 'ann_dir': '../images/annTrain1/',}) 
	#t2 = multiprocessing.Process(target=seg2ann, kwargs={'seg_dir': "../images/segTrainpng1", 'ann_dir': '../images/annTrain2/',}) 
	t3 = multiprocessing.Process(target=seg2ann, kwargs={'seg_dir': "../images/segTrainpng3", 'ann_dir': '../images/annTrain3/',}) 
	t4 = multiprocessing.Process(target=seg2ann, kwargs={'seg_dir': "../images/segTrainpng4", 'ann_dir': '../images/annTrain4/',}) 
	t5 = multiprocessing.Process(target=seg2ann, kwargs={'seg_dir': "../images/segTrainpng5", 'ann_dir': '../images/annTrain5/',}) 

    # starting process 1 
	#t1.start() 
    # starting process 2 
	#t2.start() 
	t3.start() 
	t4.start() 
	t5.start() 
	print("All process started!")
 
    # wait until process 1 is completely executed 
	#t1.join() 
    # wait until process 2 is completely executed 
	#t2.join() 
	t3.join() 
	t4.join() 
	t5.join() 
  
    # both threads completely executed 
	print("Done!")


