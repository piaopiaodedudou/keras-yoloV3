from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization,add,Activation
from keras.layers.advanced_activations import LeakyReLU
from baseModel import BaseModel
from keras.models import Model

class YoloV3(BaseModel):
    def __init__(self,input_size):
        input_image = Input(shape=(input_size, input_size, 3))
        #layer1
        x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=True)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer2
        x = Conv2D(64, (3,3), strides=(2,2), padding='same', name='conv_2', use_bias=True)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer3
        shortcut = Activation('linear')(x)
        x = Conv2D(32, (1,1), strides=(1,1), padding='same', name='conv_3', use_bias=True)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer4
        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_4', use_bias=True)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = add([x,shortcut])
        #layer5
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=True)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer6
        shortcut = Activation('linear')(x)
        x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_6', use_bias=True)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer7
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_7', use_bias=True)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = add([x,shortcut])
        #layer8
        shortcut = Activation('linear')(x)
        x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_8', use_bias=True)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer9
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=True)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = add([x,shortcut])
        #layer10
        x = Conv2D(256, (3,3), strides=(2,2), padding='same', name='conv_10', use_bias=True)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer11 - layer26
        for i in range(0,8):
            shortcut = Activation('linear')(x)
            x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_'+str(2*i+11), use_bias=True)(x)
            x = BatchNormalization(name='norm_'+str(2*i+11))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_'+str(2*i+12), use_bias=True)(x)
            x = BatchNormalization(name='norm_'+str(2*i+12))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = add([x,shortcut])
        #layer27
        x = Conv2D(512, (3,3), strides=(2,2), padding='same', name='conv_27', use_bias=True)(x)
        x = BatchNormalization(name='norm_27')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer28-layer44
        for i in range(0,8):
            shortcut = Activation('linear')(x)
            x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_'+str(2*i+28), use_bias=True)(x)
            x = BatchNormalization(name='norm_'+str(2*i+28))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_'+str(2*i+29), use_bias=True)(x)
            x = BatchNormalization(name='norm_'+str(2*i+29))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = add([x,shortcut])
        #layer45
        x = Conv2D(1024, (3,3), strides=(2,2), padding='same', name='conv_45', use_bias=True)(x)
        x = BatchNormalization(name='norm_45')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer46 - layer53
        for i in range(0,4):
            shortcut = Activation('linear')(x)
            x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_'+str(2*i+46), use_bias=True)(x)
            x = BatchNormalization(name='norm_'+str(2*i+46))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_'+str(2*i+47), use_bias=True)(x)
            x = BatchNormalization(name='norm_'+str(2*i+47))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = add([x,shortcut])
        # feature_extractor
        self.feature_extractor = Model(input_image, x, name="yolov3")

    def get_layers_info(self):
        print self.feature_extractor.summary()
        for layer in self.feature_extractor.layers:
            print("{} output shape: {}".format(layer.name, layer.output_shape))
            print layer.output

    def get_layers_feauture(self):
        return self.feature_extractor