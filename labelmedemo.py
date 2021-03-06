#!/usr/bin/env python
 
import argparse
import json
import matplotlib.pyplot as plt
 
from labelme import utils
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('0000.json')
    args = parser.parse_args()
 
    json_file = args.json_file
 
    data = json.load(open(json_file))
 
    img = utils.img_b64_to_array(data['imageData'])
    lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
 
    lbl_viz = utils.draw_label(lbl, img, lbl_names)
 
    plt.imshow(lbl_viz)
    plt.show()
 
 
# if __name__ == '__main__':
