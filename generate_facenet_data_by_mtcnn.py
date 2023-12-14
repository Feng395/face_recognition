import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from time import time

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

dataset=datasets.ImageFolder('data/photos') # photos folder path
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

# face_list = [] # list of cropped faces from photos folder
name_list = [] # list of names corrospoing to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True)
    # face, prob = yolov5.dect_img_one_output(img)
    print("第",idx,"种图片置信度为：",prob)
    if face is not None and prob>0.90: # if face detected and porbability > 90%
        emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
        embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
        name_list.append(idx_to_class[idx]) # names are stored in a list

data = [embedding_list, name_list]
torch.save(data, 'facenet2.pt') # saving data.pt file

print("共计处理",len(loader),"张图片,选择了",len(embedding_list),"张图片置信度满足条件进行留存！")

'''
第 0 张图片置信度为： 0.9999752044677734
第 0 张图片置信度为： 0.9999434947967529
第 1 张图片置信度为： 0.9976255297660828
第 2 张图片置信度为： 0.9999340772628784
第 3 张图片置信度为： 0.999816358089447
第 4 张图片置信度为： 0.9999340772628784
第 5 张图片置信度为： 0.9998761415481567
第 6 张图片置信度为： 0.9997329115867615
第 7 张图片置信度为： 0.9999922513961792
第 7 张图片置信度为： 0.9999939203262329
第 8 张图片置信度为： 0.9998761415481567
第 9 张图片置信度为： 0.9999923706054688
第 10 张图片置信度为： 0.9999923706054688
共计处理 13 张图片,选择了 13 张图片置信度满足条件进行留存！
'''