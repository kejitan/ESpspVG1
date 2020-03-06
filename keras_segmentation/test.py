import pandas as pd
import re
import json
from pandas.io.json import json_normalize

from PIL import Image 
import glob
from tqdm import tqdm
import six

import random
import os

import cv2
import numpy as np
from keras.models import load_model

from  train import find_latest_checkpoint
from  data_utils.data_loader import get_image_array, get_segmentation_array, DATA_LOADER_SEED, class_colors , get_pairs_from_paths
from  models.config import IMAGE_ORDERING
import metrics
from keras_segmentation.pretrained import pspnet_50_ADE_20K 

random.seed(DATA_LOADER_SEED)



json.load((open('/media/kejitan/Blue4TB/datasets/VisualGenome/1.2/image_data.json')))
with open('/media/kejitan/Blue4TB/datasets/VisualGenome/1.2/image_data.json') as json_string:
    image_str = json.load(json_string)

image_str_data = json_normalize(image_str)

for i in range( 0, 100) :
    url = image_str_data.at[i, 'url']

    subfields = url.split('/')
    subdir = subfields[-2]
    if subdir == 'VG_100K' :
        imagedir = 'images'
    elif subdir == 'VG_100K_2' :
        imagedir = 'images2'
    image_name = subfields[-1]
    print("image_name = {}", image_name)
    image_path = '/media/kejitan/Blue4TB/datasets/VisualGenome/1.2/'
    image_path = image_path+imagedir+'/'+subdir+'/'+image_name
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(img, (473,473), interpolation = cv2.INTER_AREA)
    cv2.imwrite('../VGdata/imgtrain100_jpg/'+image_name, resized)

json_string.close()               

in_dir = "../VGdata/imgtrain100_jpg/"
out_dir = "../VGdata/imgtrain100_png/"

for filename in os.listdir(in_dir):
    if filename.endswith(".jpg"): 
        #convert png to jpg with cv2..
        im = Image.open(in_dir+filename)
        nameext = filename.split('.')
        name = nameext[0]
        rgb_im = im.convert('RGB')
        rgb_im.save(out_dir+nameext[0]+'.png')
        im.close()
        rgb_im.close()
'''
inp = glob.glob(os.path.join(in_dir, "*.jpg")) 
for i, inp in enumerate(tqdm(inp)):
	if isinstance(inp, six.string_types):
		out_fname = os.path.basename(inp)
		file, ext = os.path.splitext(out_fname)
		im = Image.open(inp)
		rgb_im = im.convert('RGB')
		rgb_im.save(out_dir + file + ".png", "PNG")
'''
def model_from_checkpoint_path(checkpoints_path):

    from models.all_models import model_from_name
    assert (os.path.isfile(checkpoints_path+"_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(
        open(checkpoints_path+"_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (latest_weights is not None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    model.load_weights(latest_weights)
    return model


def predict(model=None, inp=None, out_fname=None, checkpoints_path=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (inp is not None)
    assert((type(inp) is np.ndarray) or isinstance(inp, six.string_types)
           ), "Inupt should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)

    seg_img = np.zeros((output_height, output_width, 3))
    colors = class_colors

    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c)*(colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if out_fname is not None:
        cv2.imwrite(out_fname, seg_img)

    return pr


def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None,
                     checkpoints_path=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    if inps is None and (inp_dir is not None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(
            os.path.join(inp_dir, "*.png")) + \
            glob.glob(os.path.join(inp_dir, "*.jpeg"))

    assert type(inps) is list

    all_prs = []

    for i, inp in enumerate(tqdm(inps)):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")

        pr = predict(model, inp, out_fname)
        all_prs.append(pr)

    return all_prs



def evaluate( model=None , inp_images=None , annotations=None,inp_images_dir=None ,annotations_dir=None , checkpoints_path=None ):
    
    if model is None:
        assert (checkpoints_path is not None) , "Please provide the model or the checkpoints_path"
        model = model_from_checkpoint_path(checkpoints_path)
        
    if inp_images is None:
        assert (inp_images_dir is not None) , "Please privide inp_images or inp_images_dir"
        assert (annotations_dir is not None) , "Please privide inp_images or inp_images_dir"
        
        paths = get_pairs_from_paths(inp_images_dir , annotations_dir )
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])
        
    assert type(inp_images) is list
    assert type(annotations) is list
        
    tp = np.zeros( model.n_classes  )
    fp = np.zeros( model.n_classes  )
    fn = np.zeros( model.n_classes  )
    n_pixels = np.zeros( model.n_classes  )
    
    for inp , ann   in tqdm( zip( inp_images , annotations )):
        pr = predict(model , inp )
        gt = get_segmentation_array( ann , model.n_classes ,  model.output_width , model.output_height , no_reshape=True  )
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()
                
        for cl_i in range(model.n_classes ):
            
            tp[ cl_i ] += np.sum( (pr == cl_i) * (gt == cl_i) )
            fp[ cl_i ] += np.sum( (pr == cl_i) * ((gt != cl_i)) )
            fn[ cl_i ] += np.sum( (pr != cl_i) * ((gt == cl_i)) )
            n_pixels[ cl_i ] += np.sum( gt == cl_i  )
            
    cl_wise_score = tp / ( tp + fp + fn + 0.000000000001 )
    n_pixels_norm = n_pixels /  np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)
    return {"frequency_weighted_IU":frequency_weighted_IU , "mean_IU":mean_IU , "class_wise_IU":cl_wise_score }

predict_multiple(model=pspnet_50_ADE_20K(), inp_dir='../VGdata/imgtrain100_png', out_dir='../VGdata/anntrain100_png', checkpoints_path='./checkpoints')

#predict_multiple(model='pspnet_50', inp_dir='VGdata/imgtrain', out_dir='VGdata/anntrain')

from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.models.pspnet import pspnet_50

pretrained_model = pspnet_50_ADE_20K()

keji_model1 = pspnet_50( n_classes=150 )

transfer_weights( pretrained_model , keji_model1  ) # transfer weights from pre-trained model to your model

keji_model1.train(
    train_images =  "../VGdata/imgtrain100_png/",
    train_annotations = "../VGdata/anntrain100_png/",
    checkpoints_path = "./checkpoints" , epochs=1
#    checkpoints_path = "./checkpoints/" , epochs=1, verify_dataset = False
)


