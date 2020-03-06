from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.models.pspnet import pspnet_50

pretrained_model = pspnet_50_ADE_20K()

keji_model1 = pspnet_50( n_classes=150 )

transfer_weights( pretrained_model , keji_model1  ) # transfer weights from pre-trained model to your model

keji_model1.train(
    train_images =  "../VGdata/images_prepped_train_png/",
    train_annotations = "../VGdata/annotations_prepped_train_png/",
    checkpoints_path = "./keji1check" , epochs=5
)

