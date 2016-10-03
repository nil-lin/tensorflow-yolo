"""
convert voc2007 data to several binary files for tensorflow read
"""
import os
import xml.etree.ElementTree as ET 
import struct
import cv2
import numpy as np

classes_name = ['person', 'bird', 'cat', 'cow', 'dog',
    'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus',
    'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable',
    'pottedplant', 'sofa', 'tvmonitor']

classes_num = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5,
    'sheep': 6, 'aeroplane': 7, 'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11,
    'motorbike': 12, 'train': 13, 'bottle': 14, 'chair': 15, 'diningtable': 16,
    'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19}

YOLO_ROOT = os.path.abspath('../')
DATA_PATH = os.path.join(YOLO_ROOT, 'data/VOCdevkit2007')
NUM_PER_BINARY = 900
IMAGE_HEIGHT = 448
IMAGE_WIDTH = 448
MAX_OBJECT_PER_IMAGE = 20


def convert_to_bytes(image, labels):
  """convert single image and labels to bytes array

  Args:
    image: ndarray[IMAGE_HEIGHT, IMAGE_WIDTH, 3]
    labels: list of (class_label, xmin, ymin, xmax, ymax)

  Returns:
    results: bytes object
  """
  results = image.tobytes()
  for label in labels:
    results += struct.pack('iiiii', *label)

  return results

def parse_xml(xml_file):
  """parse xml_file

  Args:
    xml_file: the input xml file path

  Returns:
    image: ndarray[IMAGE_HEIGHT, IMAGE_WIDTH, 3]
    labels: list of (class_label, xmin, ymin, xmax, ymax)
  """
  tree = ET.parse(xml_file)
  root = tree.getroot()

  image = None
  labels = []
  for item in root:
    if item.tag == 'filename':
      image_path = os.path.join(DATA_PATH, 'VOC2007/JPEGImages', item.text)
    elif item.tag == 'size':
      width = int(item.find('width').text)
      height = int(item.find('height').text)
      depth = int(item.find('depth').text)
      width_rate = IMAGE_WIDTH * 1.0 / width
      height_rate = IMAGE_HEIGHT * 1.0 / height 
    elif item.tag == 'object':
      obj_name = item[0].text
      obj_num = classes_num[obj_name]
      xmin = int(int(item[4][0].text) * width_rate)
      ymin = int(int(item[4][1].text) * height_rate)
      xmax = int(int(item[4][2].text) * width_rate)
      ymax = int(int(item[4][3].text) * height_rate)
      center_x = int((xmin + xmax)/2)
      center_y = int((ymin + ymax)/2)
      w = xmax - xmin
      h = ymax - ymin
      labels.append([obj_num, center_x, center_y, w, h])


  image = cv2.imread(image_path)
  image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

  total_object = len(labels)

  if total_object < MAX_OBJECT_PER_IMAGE:
    labels += [[-1,-1,-1,-1,-1]] * (MAX_OBJECT_PER_IMAGE - total_object)

  labels = labels[:MAX_OBJECT_PER_IMAGE][:]


  return image, labels



def write_to_binary(xml_files, out_file):
  """write several single image and labels to binary file

  Args:
    xml_files: list of xml_file names: list(str)
    out_file : output binary file path: str
  
  Returns:
    nothing
  """
  global record_number
  out_file = open(out_file, 'wb')

  for xml_file in xml_files:
    try:
      image, labels = parse_xml(xml_file)
      results = convert_to_bytes(image, labels)
      out_file.write(results)
    except Exception:
      pass
  out_file.close()


def main():
  out_path = os.path.join(YOLO_ROOT, 'data/voc_binary_data')
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  xml_dir = DATA_PATH + '/VOC2007/Annotations/'

  num = 0
  xml_list = os.listdir(xml_dir)
  xml_list = [xml_dir + temp for temp in xml_list]
  bin_num = len(xml_list) / NUM_PER_BINARY + 1

  for i in xrange(bin_num):
    xml_files = xml_list[i * NUM_PER_BINARY : min((i + 1) * NUM_PER_BINARY, len(xml_list))]
    out_file = os.path.join(out_path, str(i) + '.bin')
    write_to_binary(xml_files, out_file)


if __name__ == '__main__':
  main()