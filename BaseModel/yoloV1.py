from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from baseModel import BaseModel

class YoloV1(BaseModel):
    def __init__(self,input_size):
        # tensorflow format :(None,w,h,c)
        input_image = Input(shape=(input_size, input_size, 3))
        #layer1
        x = Conv2D(64, (7,7), strides=(2,2), padding='same', name='conv_1', use_bias=True)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)
        #layer2
        x = Conv2D(192, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=True)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)
        #layer3
        x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_3', use_bias=True)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer4
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_4', use_bias=True)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer5
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_5', use_bias=True)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer6
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=True)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)
        #layer7 - layer14
        for i in range(0,4):
            x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_'+str(2*i+7), use_bias=True)(x)
            x = BatchNormalization(name='norm_'+str(2*i+7))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_'+str(2*i+8), use_bias=True)(x)
            x = BatchNormalization(name='norm_'+str(2*i+8))(x)
            x = LeakyReLU(alpha=0.1)(x)
        # layer15
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=True)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer16
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=True)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)
        # layer17
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=True)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer18
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=True)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)
        # layer19
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_19', use_bias=True)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer20
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=True)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)
        # layer21
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_21', use_bias=True)(x)
        x = BatchNormalization(name='norm_21')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer22
        x = Conv2D(1024, (3,3), strides=(2,2), padding='same', name='conv_22', use_bias=True)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)
        # layer23
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_23', use_bias=True)(x)
        x = BatchNormalization(name='norm_23')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer24
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_24', use_bias=True)(x)
        x = BatchNormalization(name='norm_24')(x)
        x = LeakyReLU(alpha=0.1)(x)
        # feauture_extractor
        self.feature_extractor = Model(input_image, x,name="yolov1")
    def get_layers_info(self):
        print self.feature_extractor.summary()
        for layer in self.feature_extractor.layers:
            print("{} output shape: {}".format(layer.name, layer.output_shape))
            print layer.output

    def get_layers_feauture(self):
        return self.feature_extractor
