#!/usr/bin/env python
import sys
import argparse
import tensorflow as tf
import cv2
from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader
import requests
from download import safe_download
import numpy as np
import hashlib
import os
from PIL import Image
import random
IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"
path='temp_image'

def faked_call_4_urls(num=10):
    #arr = ["http://img.mxtrip.cn/fadd1b80f8f62eb335cca0a1ffb777f1.jpeg"]
    urls=[]
    f = open("urls.txt") 
    line = f.readline()  
    while line:
        urls.append(line.replace('\n',''))
        if len(urls) == 10:
            print(line)
            yield urls
            urls=[]
        line = f.readline()
    f.close()
    yield urls

file = "detect_result.txt"

def write_to_file(result):   
    with open(file, 'a+') as f:
        f.write(result + "\n")
    f.close()
def max_prob(predictions,size=20):
    prob = 0.0
    for i in range(20):
        if prob < predictions[i][1]:
            prob = predictions[i][1]
    return prob
    
def read_frame_from_video(videopath):
    video_capture = cv2.VideoCapture(videopath)
    images = []
    while True:
        ret, frame = video_capture.read()
        if ret:
            image = Image.fromarray(frame)
      #      print("before",frame.shape)
            image = image.resize((224, 224))
            images.append(np.array(image))
        else:
            break
    
    tot = len(images)

    print("total images from video",videopath,len(images))
   
    if tot > 20:
        return random.sample(images,20)
       
    return images

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file",default="urls.txt", help="Path to the input image.\
                        Only jpeg images are supported.")
    parser.add_argument("-m", "--model_weights", required=True,
                        help="Path to trained model weights file")

    parser.add_argument("-l", "--image_loader",
                        default=IMAGE_LOADER_YAHOO,
                        help="image loading mechanism",
                        choices=[IMAGE_LOADER_YAHOO, IMAGE_LOADER_TENSORFLOW])

    parser.add_argument("-t", "--input_type",
                        default=InputType.TENSOR.name.lower(),
                        help="input type",
                        choices=[InputType.TENSOR.name.lower(),
                                 InputType.BASE64_JPEG.name.lower()])

    args = parser.parse_args()

    model = OpenNsfwModel()

    with tf.Session() as sess:

        input_type = InputType[args.input_type.upper()]
        model.build(weights_path=args.model_weights, input_type=input_type)

        fn_load_image = None

        if input_type == InputType.TENSOR:
            if args.image_loader == IMAGE_LOADER_TENSORFLOW:
                fn_load_image = create_tensorflow_image_loader(sess)
            else:
                fn_load_image = create_yahoo_image_loader()
        elif input_type == InputType.BASE64_JPEG:
            import base64
            fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])
      
        sess.run(tf.global_variables_initializer())
        urls_gen = faked_call_4_urls()
        for urls in urls_gen:
            for url in urls:
                md5=hashlib.md5(url.encode('utf-8')).hexdigest()
                name = path+"/"+md5+'.jpg';
                if os.path.exists(name) == False :
                    print("begin to download imgurl=",url)
                    name = safe_download("temp_image",url)
                else:
                    print("exist file name="+name)
                image = read_frame_from_video(name)
                print("need to detect image size ",len(image)) 
                predictions = sess.run(model.predictions,feed_dict={model.input: image})
                write_to_file(url+"\t"+str(max_prob(predictions)))
                print(url+"\t"+str(max_prob(predictions)))
                if os.path.exists(name) == True :
                    os.remove(name)
                    print("delete file name="+name)
if __name__ == "__main__":
    main(sys.argv)
