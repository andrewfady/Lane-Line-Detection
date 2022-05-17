import cv2
import torch
import time
import warnings
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np

model = torch.hub.load('ultralytics/yolov5', "yolov5m")
debug=False


def cut_image_api(image,array_positions):
    array_photos=[]
    for single_record in array_positions:

        xmin = int(single_record[0])

        ymin = int(single_record[1])

        xmax = int(single_record[2])

        ymax = int(single_record[3])

        im1 = image[ ymin:ymax,xmin:xmax,:3]

        array_photos.append(im1)

    return array_photos




def pipline_outputs_combined(main_frame,stages_output):
    #Get the orginal frame height
    h1, w1 = main_frame.shape[:2]
    #Get pipeline output images size
    square_size=int(w1/len(stages_output))        
    img_3 = np.zeros((h1,w1,3), dtype=np.uint8)
    img_3[:h1, :w1,:3] = main_frame
    i=0
    #Add pipeline output images to the original frame 
    for image in stages_output:
        image=cv2.resize(image,(square_size,200),interpolation = cv2.INTER_AREA)
        img_3[0:200,i*square_size:(i*square_size)+square_size,:3] = image
        i+=1
    return img_3




def draw_prediction(img, array_out):
    for single_out in array_out:
        if (single_out[6]=='car'):
            color = (250,0,0)
            point1=(int(single_out[0]),int(single_out[1]))
            point2=(int(single_out[2]),int(single_out[3]))
            cv2.rectangle(img,point1, point2, color, 2)



def detect_image(image_cv2):
    results = model(image_cv2)
    image_cv1 = image_cv2.copy()
    array_out = results.pandas().xyxy[0].to_numpy()
    cropped_car_images=cut_image_api(image_cv2,array_out)
    draw_prediction(image_cv1, array_out)
    if debug and not(len(cropped_car_images)==0):
        return pipline_outputs_combined(image_cv1,cropped_car_images)
    return image_cv1


import sys
input=sys.argv
challenge_output =input[2]
clip1 = VideoFileClip(input[1])
debug=int(input[3])
challenge_clip = clip1.fl_image(detect_image)
challenge_clip.write_videofile(challenge_output, audio=False)