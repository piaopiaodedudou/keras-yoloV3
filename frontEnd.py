from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
import numpy as np
import os
import cv2
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

class Yolo3(object):
    def __init__(self,backend,
                    input_size,
                    labels,
                    anchors,
                 max_box_per_image):
        self.input_size =input_size
        self.labels  = labels
        self.bndclass = len(labels)
        self.anchors = anchors
        self.box = len(anchors)/2
        self.max_box_per_image = max_box_per_image
        input_image = Input(shape=(self.input_size,self.input_size,3))
        true_boxes = Input(shape=(1,1,1,max_box_per_image,4))

        #define the backend of the yolo3
        if backend == 'yolov3':
            from BaseModel.yoloV3 import YoloV3
            yolo3 = YoloV3(input_size)
            self.feature_extractor = yolo3.get_layers_feature()
            self.extractor_output = yolo3.extractor_output(input_image)
            output1 = self.extractor_output[0]
            output2 = self.extractor_output[1]
            output3 = self.extractor_output[2]

        ######################################

        output1 = Conv2D(self.box *(4+1+self.bndclass),
                        (1,1),strides =(1,1),
                        padding='same',name ='DetectionLayer_output1'
                        )(output1)
        output1 = Reshape((output1.shape[1].value,output1.shape[1].value,self.box,4+1+self.bndclass))(output1)

        output2 = Conv2D(self.box *(4+1+self.bndclass),
                        (1,1),strides =(1,1),
                        padding='same',name ='DetectionLayer_output2'
                        )(output2)
        output2 = Reshape((output2.shape[1].value,output2.shape[1].value,self.box,4+1+self.bndclass))(output2)

        output3 = Conv2D(self.box *(4+1+self.bndclass),
                        (1,1),strides =(1,1),
                        padding='same',name ='DetectionLayer_output3'
                        )(output3)
        output3 = Reshape((output3.shape[1].value,output3.shape[1].value,self.box,4+1+self.bndclass))(output3)

        output = [output1,output2,output3]
        #output = Lambda(lambda args: args[1])([output, self.true_boxes])
        self.model = Model([input_image,true_boxes],output)
        print self.model.summary()

# test
yolov3  =Yolo3('yolov3',416,['w']*20,[1,2,3,4,5,6],10)

# input_image = Input(shape=(416, 416, 3))
# true_boxes = Input(shape=(1, 1, 1, 10, 4))
# print yolov3.model([input_image,true_boxes])


