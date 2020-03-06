#README.md

1. Install ElasticSearch7.4.2 or above

2. sudo apt-get install -y libsm6 libxext6 libxrender-dev

3. 
```
git clone https://github.com/kejitan/ESpspVG1
cd ESpspVG1
--Create conda virtual environment PSP by running:
conda create -n PSP python=3.6
conda activate PSP
pip install -r requirements.txt 
```

4. Image query based part of the project uses Visual Genome dataset from http://visualgenome.org/api/v0/api_home.html with Image meta data annotations. 
Text query based part uses Visual Genome Data set with Image meta data and Object annotations. 
we will be downloading version 1.2 dataset parts 1 and 2, together with annotations 'Image meta data and Objects into ESpspVG1 directory. We should be having images.zip, images2.zip, image_data.json.zip, objects_v1_2.json.zip in the ESpspVG1 directory

5. Uncompress all zip files. we will have images, images2 directory, image_data.json file. Move object.json file from objects_v1_2.json directory to
ESpspVG1 directory. 

6. The images are in jpg format. Resize the images to 473x473 format by running the following in ESpspVG1 directory 
```
mkdir -p assets
python makepsp150data.py
```
The image segmantation and annotations step requires that the images be in 473x473 PNG format). 

7. The images in assets directory need to be segmented and annotated. That procedure will be described in the next submission. For now the annotated files have been made available in the repository as ANN.tgz. This needs to be decompressed and extracted in ESpspVG1 directory. 

8. Next we need to create Elastic Search index on the machine on which this application is installed. This is done by Running
python ESANN.py
This will take more than an hour depending on the machine.

9. You can verify that the index 'vgnum' is created by installing Kibana and exploring Create Index in Management Tool and then browsing it in the Discover tool.

* IMAGE QUERY BASED IMAGE RETRIEVAL

10. Now we are ready to run the Image based query application. Run
python VGdashPSP.py 
from a terminal in the ESpspVG1 directory.

11. It will inform that the application will interact on localhost:8050 port. Pleasee open this port in a web browser. Click refresh button in the web browser. You will see five boxes. Top Input box and FETCHG IMAGES and CLEAR IMAGES buttons are for Query bases image search.

12. Now we need to supply the application with a sample image. This can be done by selecting a sample jpg file or dragging it on the Drag and Drop of Select Files component. Once the image is loaded, press DISPLAY SIMILAR IMAGES button. 

12. After a few seconds, you will see some informational messages in the terminal and a prompt 'Please hit refresh'. At this point the images are ready to be displayed on the web browser. We need to press Refresh button on the web browser. You will see up to 4 images that have similarity to the sample image supplied earlier. 

13. We can repeat the procedure to assess the quality of similarity of the images. This is at present not very good, since the number classes (type of objects in the images) is 150 (small). These classes can be examined in the file PSPindexClass.csv. We have 108077 images and each image has up to 11 classes identified in the images. For similarity we match top 6 classes. Still there are lots of matches and 4 images are selected randomly. Right now we have not impemented a similarity score. 


* TEXT QUERY BASED IMAGE RETRIEVAL

14. We enter comma separated list of objects we wish to see in the images that the system retrieves, in the top Input box, e.g. man,tiger 
The system will search the Visual Genome dataset using ElasticSearch and present up to 4 qualifying images. 

15. After entering the comma separated list, press FETCH IMAGES button and wait for informationl message in the terminal to Hit refresh button. After hitting the Refresh button on the browser, the browser display will be updates. 

16. After this you can present the system with another query. 

* NOTE
This is a scaled up version of the project. Jupyter notebooks have not been provided as the performance of the system is not good with small number of images in te dataset. 

* In the next submission, we will present how to add ADE20K data set to the assets as well as show the process of segmentation and annotation of original database and then subsequent addtion to the Elastic Search indexes based on our custom annotation files. (the ANN.tgz annotation files is also cusom one built by us from the raw Visual Genome Data)

