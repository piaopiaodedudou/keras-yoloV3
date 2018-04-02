from imgaug import augmenters as iaa
import xml.etree.ElementTree as ET
import os

'''
return  the train_data,train_labels, valid_data, valid_labels
'''
def parse_annotation(ann_dir, img_dir, labels=[],division=7):
    train_imgs = []
    valid_imgs = []
    seen_train_labels = {}
    seen_valid_labels = {}
    count = 1
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}

        tree = ET.parse(ann_dir + ann)

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        if (count % division == 0) :
                            if obj['name'] in seen_valid_labels:
                                seen_valid_labels[obj['name']] += 1
                            else:
                                seen_valid_labels[obj['name']] = 1

                            if len(labels) > 0 and obj['name'] not in labels:
                                break
                            else:
                                img['object'] += [obj]
                        else :
                            if obj['name'] in seen_train_labels:
                                seen_train_labels[obj['name']] += 1
                            else:
                                seen_train_labels[obj['name']] = 1

                            if len(labels) > 0 and obj['name'] not in labels:
                                break
                            else:
                                img['object'] += [obj]


                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))
        if (count % division == 0):
            if len(img['object']) > 0:
                valid_imgs += [img]
        else:
            if len(img['object']) > 0:
                train_imgs += [img]
        count += 1
    return train_imgs, seen_train_labels, valid_imgs,seen_valid_labels