from BaseModel.yoloV1 import  YoloV1
from BaseModel.yoloV1Tiny import YoloV1Tiny
from BaseModel.yoloV2 import YoloV2
from BaseModel.yoloV2Tiny import  YoloV2Tiny
from BaseModel.yoloV3 import  YoloV3
from data_util import parse_annotation
import cPickle as pickle
from keras.layers import Input
input_size = 416

input_image = Input(shape=(input_size, input_size, 3))
yolo = YoloV3(input_size)

print type(yolo.extractor_output(input_image))
print yolo.extractor_output(input_image)[0]
#train_data,train_labels,valid_data,valid_labels =parse_annotation(
#                '/data4/sunsiyuan/data/VOC/VOCdevkit/VOC2012/Annotations/',
#                '/data4/sunsiyuan/data/VOC/VOCdevkit/VOC2012/JPEGImages/'
#                                                                )

#print len(train_data)
#print train_data[0]

#print '***********************'
#print  len(valid_data)
#print  valid_data[0]

#print '***************************'
#print train_labels
#print valid_labels

# f = open('./train_data.pkl','w')
# pickle.dump(train_data,f)
# f.close()
# f = open('./valid_data.pkl','w')
# pickle.dump(valid_data,f)
# f.close()
# f = open('./train_labels.pkl','w')
# pickle.dump(train_labels,f)
# f.close()
# f = open('./valid_labels.pkl','w')
# pickle.dump(valid_labels,f)
# f.close()
# f = open('./valid_labels.pkl','r')
# data = pickle.load(f)
# print  data
# f.close()
# f = open('./valid_data.pkl','r')
# data = pickle.load(f)
# print  data
