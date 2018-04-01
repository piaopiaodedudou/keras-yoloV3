'''
implement different feature extractors ,including:

1.yolo-tiny
2.yolo
3.yolov2-tiny
4.yolov2
5.yolov3

'''

from  yoloV1 import  YoloV1
from yoloV1Tiny import YoloV1Tiny
from yoloV2 import YoloV2
from yoloV2Tiny import  YoloV2Tiny
from yoloV3 import  YoloV3

def get_feature_extractor(name,image_size):
    if name =='yolov1tiny':
        return YoloV1Tiny(image_size)
    if name =='yolov1':
        return YoloV1(image_size)
    if name == 'yolov2tiny':
        return YoloV2Tiny(image_size)
    if name =='yolov2':
        return YoloV2(image_size)
    if name =='yolov3':
        return YoloV3(image_size)
