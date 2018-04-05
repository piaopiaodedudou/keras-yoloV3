from keras.layers import  Conv2D, Input, MaxPooling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from baseModel import BaseModel
from keras.models import Model

class YoloV1Tiny(BaseModel):
    def __init__(self,input_size):
        # tensorflow format :(None,w,h,c)
        input_image = Input(shape=(input_size, input_size, 3))
        #layer1
        x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=True)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)
        #layer2
        x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=True)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)
        #layer3
        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=True)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)
        #layer4
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_4', use_bias=True)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)
        #layer5
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=True)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)
        #layer6
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=True)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer7
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_7', use_bias=True)(x)
        x = BatchNormalization(name='norm_7')(x)
        # feature_extractor
        self.feature_extractor = Model(input_image, x,name="yolov1Tiny")
    def get_layers_info(self):
        print self.feature_extractor.summary()
        for layer in self.feature_extractor.layers:
            print("{} output shape: {}".format(layer.name, layer.output_shape))
            print layer.output

    def get_layers_feauture(self):
        return self.feature_extractor

    def extractor_output(self,input_image):
        return self.feature_extractor(input_image)