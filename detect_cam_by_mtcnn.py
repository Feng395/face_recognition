import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
from time import time


mtcnn = MTCNN(image_size=240, margin=0,keep_all=True, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

# Using webcam recognize face

# loading data.pt file
load_data = torch.load('data.pt')
embedding_list = load_data[0]
name_list = load_data[1]

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("fail to grab frame, try again")
        break

    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True)

    if img_cropped_list is not None:
        '''
        mtcnn.detect(img)将返回一个包含人脸检测结果的列表。每个检测结果是一个字典，包含以下信息：
          'box': 人脸框的坐标，表示为一个包含四个整数值的列表 [x1, y1, x2, y2]，其中 (x1, y1) 是人脸框的左上角坐标，(x2, y2) 是人脸框的右下角坐标。
          'confidence': 人脸检测的置信度，表示为一个浮点数，表示模型对该人脸框的置信程度。
          'keypoints': 人脸关键点的坐标，表示为一个包含五个元组 (x, y) 的列表，分别代表眼睛中心、鼻子、左嘴角和右嘴角的位置。
        '''
        boxes, _ = mtcnn.detect(img)

        for i, prob in enumerate(prob_list):
            if prob > 0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                dist_list = []  # list of matched distances, minimum distance is used to identify the person

                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list)  # get minumum dist value
                min_dist_idx = dist_list.index(min_dist)  # get minumum dist index
                name = name_list[min_dist_idx]  # get name corrosponding to minimum dist

                box = boxes[i] # [286.9894104003906,82.09547424316406,429.5582580566406,257.08148193359375]
                '''
                'box': 人脸框的坐标，表示为一个包含四个整数值的列表 [x1, y1, x2, y2]，其中 (x1, y1) 是人脸框的左上角坐标，(x2, y2) 是人脸框的右下角坐标。
                '''
                pt1 = (int(box[0]), int(box[1]))  # 左上角坐标
                pt2 = (int(box[2]), int(box[3]))  # 右下角坐标

                original_frame = frame.copy()  # storing copy of frame before drawing on it

                if min_dist < 0.90:
                    frame = cv2.putText(frame, name + ' ' + str(round(min_dist, 2)), pt1, cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 255, 0), 1, cv2.LINE_AA,False)

                frame = cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)

    cv2.imshow("IMG", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC
        print('Esc pressed, closing...')
        break

    elif k % 256 == 32:  # space to save image
        print('Enter your name :')
        name = input()

        # create directory if not exists
        if not os.path.exists('photos/' + name):
            os.mkdir('photos/' + name)

        img_name = "photos/{}/{}.jpg".format(name, int(time()))
        cv2.imwrite(img_name, original_frame)
        print(" saved: {}".format(img_name))

cam.release()
cv2.destroyAllWindows()
