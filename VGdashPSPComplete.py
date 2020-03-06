import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from textwrap import dedent
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_player as player

import random
import plotly
import plotly.graph_objs as go
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate

import numpy as np
import pandas as pd

import pathlib
import PIL

import gc
import json
import requests
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import bulk
from elasticsearch_dsl.query import Bool, MultiMatch, Q
from elasticsearch_dsl.search import Search, MultiSearch
from elasticsearch_dsl import Mapping, Keyword, Nested, Text
from elasticsearch_dsl import Index, analyzer, tokenizer
import glob
import cv2
from matplotlib import pyplot as plt
from findClassANN import find_classes
from PIL import Image
import base64
import io
import string
#import os


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
app.config.suppress_callback_exceptions = True

res = requests.get('http://localhost:9200')
print (res.content)
es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
client = Elasticsearch()
#app.run_server(host='127.0.0.1', port=8053)

def main():
    init_layout(1_000)
    app.run_server(host='127.0.0.1')

def init_layout(refresh_interval):
    app.layout = serve_layout([])

def query_imagesi(classnum_list):
    print("In query_imagesi")
    hit1 = set()
    image_set = set()
    print("11. classnum_list =", classnum_list)
    
    QI = Q('match_all')
    s1 = Search(index='vgnum')
    #s1 = Search(index='tryann1')
    classn = 1
    for class_num in classnum_list:
        if classn > 6 :
            break
        classn = classn + 1
        print("class_num= ",class_num)
        QI = QI & Q('bool', must=[Q("match", classnum=class_num)])

    s1 = s1.query(QI).using(client)
    response = s1.execute()
    #print(response)
    for hit in s1.scan() :
        #print("33 ", hit.imgfile)
        image_set.add(hit.imgfile)
    display_image_set(image_set)


    
def query_imageso(object_list):
    print("In query_imageso")
    hit1 = set()
    image_set = set()
    print("11. object_list =", object_list)

    
    QI = Q('match_all')
    s1 = Search(index='idx0')
    for name in object_list:
        print("name= ",name)
        QI = QI & Q("match", names=name)

    s1 = s1.query(QI).using(client)
    response = s1.execute()
    #print(response)
    for hit in s1.scan() :
        #print("33 ", hit.imgfile)
        image_set.add(hit.imgfile)

    display_image_set(image_set)


def display_image_set(image_set) :

    #print("image_set = {0}".format(image_set))
    im = 0
    #app.layout = serve_layout
    images_div = []
    for image in image_set :
        if im > 3 : 
            break
        file, ext = os.path.splitext(image)
        image = file + '.jpg'
        print("66 image =", image)
        images_div.append(display_image(image))
        im = im + 1
    print("Please hit refresh...")
    # Here call callback -
    #serve_layout = 
    images_div.append(html.Div(id='output-images' ))
    app.layout = serve_layout(images_div)

    return



def no_images_msg():
    return html.div([
            html.Output(id='no_images', value="No images found")
    ])

def display_image(image):
    return html.Div(
        html.A(
            html.Img(
                src = app.get_asset_url(image)#,
				#style={'display':'block'}
           ) )
    )



def serve_layout(img_div):
    return html.Div(    
		children=[
		    dcc.Interval(id="interval-updating-images", interval=100000, n_intervals=0),
		    html.Div(
		        className="container",
		        children=[
		            html.Div(
		                id="left-side-column",
		                className="eight columns",
		                children=[
		                    html.Img(
		                        id="logo-mobile", src=app.get_asset_url("dash-logo.png")
		                    ),
		                    html.Label('Objects in Image'),
		                    html.Div([
		                        html.Div(dcc.Input(id="Objects-in-image", value="man",type='text')),                       
		                        html.Button( children="Fetch Images", id="fetch-images",  n_clicks=0),
		                        html.Div(id='outputf', children="fimage"),
		                        html.Button( children="Clear Images", id="clear-images", n_clicks=0),
								html.Div(id='display-clear-button', children="dimage"),
		                        html.Div(img_div, id='disp-images' ),

		                    ]),
                            dcc.Upload(
                                id='upload-image',
                                children=html.Div([
                                    'Drag and Drop or ',
                                     html.A('Select Files')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
 
                                # Allow multiple files to be uploaded
                                multiple=False
                                ),
                                html.Div(id='output-image-upload'),
                                html.Div(id='output-similar-images' ),
		                        html.Button( children="Display Similar Images", id="display-similar-images",  n_clicks=0),
                                #html.Div(id='display-similar-images' ),
                                html.Div(id='output-images' ),



		                ],
		            ),
		        ],
		    ),
		]
)


app.layout = serve_layout([])

@app.callback(Output('outputf', 'children'),
             [Input('fetch-images', 'n_clicks')],
              [State('Objects-in-image', 'value')])
def fetch_images(n_clicks, value):
    if n_clicks > 0:
        n_clicks = 0
        #print("value=", value)
        object_list = value.split(',')
        print("22. object_list=",object_list)
        query_imageso(object_list)


@app.callback(Output('display-clear-button', 'children'),
             [Input('clear-images', 'n_clicks')] )
def clear_images(n_clicks):
    if n_clicks > 0:
        n_clicks = 0
        #images_div = []
        app.layout = serve_layout([])
        print("In clear_images")


def parse_contents(contents, filename):
    print("file_name = ", filename)
    try:
        fname = os.path.basename(filename)
        file, ext = os.path.splitext(fname)

        if ( ext in ['jpg', 'JPG', 'JPEG', 'png'] ):
        # Assume that the user uploaded an image file
            dummy = True;
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    content_type, content_string = contents.split(',')
    image = base64.b64decode(content_string) #.convert('RGB')
    image = Image.open(io.BytesIO(image))
    rgb_im = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_im, (473,473), interpolation = cv2.INTER_AREA)
    cv2.imwrite("/var/tmp/"+file+".png", resized)
    print("111"+ " /var/tmp/"+file+".png")
    return html.Div([
        html.H5(filename),
        html.Img(src=contents),
        html.Hr()  # horizontal line        
    ])


@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename')])
def update_sample(contents, file_name):
    if contents is not None:
        print("222 " + file_name)
        children = [ parse_contents(contents, file_name) ]
        return children


@app.callback(Output('output-similar-images', 'children'),
             [Input('display-similar-images', 'n_clicks')],
             #[Input('upload-image', 'contents')],
             [State('upload-image', 'filename')] )
def display_similar_images( n_clicks, filename ): # image in jpg or mpg format
    if n_clicks > 0:
        print("click_received")
        n_clicks = 0

        fname = os.path.basename(filename)
        file, ext = os.path.splitext(fname)

        CDict = find_classes("/var/tmp/"+file+".png", "/var/tmp/"+file+"seg.png")
        #print("4444 CDict" )
        #print(CDict)	
        query_imagesi(CDict)
 


def show_images(contents, filename):
    try:
        if ( ('jpg' in filename) or ('JPG' in filename) or ('png' in filename) or ('PNG' in filename) ):
        # Assume that the user uploaded an image file
            dummy = True;
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.Img(src=contents),
        html.Hr(),  # horizontal line        
    ])

if __name__ == "__main__":
    main()
#    app.run_server(debug=True, port=8053)
