from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm

#######################################################################################
# create new folder and put .data and .names and .cfg and .weights files into folder

folder="autotag_dataset"  ### path of new images wants to be tag
cfg_file= "autotag_model/yolov4.cfg"
data_file="autotag_model/obj.data"
weight_file="autotag_model/yolov4.weights"

######################################################################################



# import darknet functions to perform object detections
from darknet import *
# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network(cfg_file, data_file, weight_file)
width = network_width(network)
height = network_height(network)
#print(class_names)
# darknet helper function to run detection on image
def darknet_helper(img, width, height):
  darknet_image = make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

  # get image ratios to convert bounding boxes to proper size
  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  # run model on darknet style image to get detections
  copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = detect_image(network, class_names, darknet_image)
  free_image(darknet_image)
  return detections, width_ratio, height_ratio
  
########################################################################################
 
def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]
 
#######################################################################################  
  



im=glob.glob(folder+"/*.jpg")
for i in tqdm(im):
    image = cv2.imread(i)
    file_name = os.path.basename(i)[:-4]
    wid, hig, _ = image.shape
    detections, width_ratio, height_ratio = darknet_helper(image, wid, hig)
    
    for label, confidence, bbox in detections:
      x1,y1,x2,y2 = bbox2points(bbox)
      bb = pascal_voc_to_yolo(x1,y1,x2,y2,wid,hig)
      #print(bb)

      with open("./"+folder+"/"+file_name+'.txt', 'a') as f:
            for j in class_names:
                if label==j:
                    obj_id=class_names.index(j)
                    f.write(str(obj_id)+' '+str(bb[0])+' '+str(bb[1])+' '+str(bb[2])+' '+str(bb[3])+'\n')
                    f.close()
        
print("##########################################################################################")  
print("DONE THANKS. :)")
print("##########################################################################################")  

#################################################################################################
