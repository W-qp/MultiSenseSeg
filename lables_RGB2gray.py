# TODO:Convert labels in RGB format to 8-bit grayscale format
import glob
import json
import os
import cv2
import numpy as np

# args
input_path = r".../*png"  # RGB labels path
save_path = r".../gray_label/"  # gray labels path
json_label_path = "json/classes.json"

assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
json_file = open(json_label_path, 'r')
color = json.load(json_file)


def rgb2gray(input_img, save_img):
    b, g, r = cv2.split(cv2.imread(input_img))
    img = cv2.merge([r, g, b])
    h, w, c = img.shape
    gray_img = np.zeros([h, w])
    for i, v in enumerate(color.values()):
        idx = np.where((img[..., 0] == v[0]) & (img[..., 1] == v[1]) & (img[..., 2] == v[2]))
        gray_img[idx] = i
    cv2.imwrite(save_img + input_img.split('\\')[-1].split('.')[0] + '.png', gray_img)


labels = glob.glob(input_path)
for label in labels:
    rgb2gray(label, save_path)
    print('{} conversion completed!'.format(label.split('\\')[-1]))
print('\nover')
