'''
图片识别只识别置信度最高的那张人脸
'''
import argparse

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


def face_match(img_path, data_path):
    '''
    img_path= location of photo, data_path= location of data.pt
    '''
    # getting embedding matrix of the given img
    img = Image.open(img_path)
    face, prob = mtcnn(img, return_prob=True)  # returns cropped face and probability

    # 调整通道顺序
    emb = resnet(face.unsqueeze(0)).detach()  # detech is to make required gradient false

    saved_data = torch.load(data_path)  # loading data.pt file
    embedding_list = saved_data[0]  # getting embedding data
    name_list = saved_data[1]  # getting list of names
    dist_list = []  # list of matched distances, minimum distance is used to identify the person

    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)

    idx_min = dist_list.index(min(dist_list))
    return (name_list[idx_min], min(dist_list), prob)


def parse_opt(known=False):
    # 创建解析器对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数
    parser.add_argument('--img_path', type=str, help='Your image path')
    # parser.add_argument('--age', type=int, help='Your age')
    # parser.add_argument('--gender', choices=['male', 'female'], help='Your gender')

    # 解析命令行参数
    args = parser.parse_args()
    return args

# img_path = 'data/test_img/many_people.jpg'
# result = face_match(img_path, 'data.pt')
#
# print('脸部识别结果为: ', result[0], 'With distance: ', result[1], 'With 检测到人脸置信度: ', result[2])
#

def main(opt):
    result = face_match(opt.img_path, 'data.pt')

    print('脸部识别结果为: ', result[0], 'With distance: ', result[1], 'With 检测到人脸置信度: ', result[2])

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

'''
命令行输入：
    python detect_img_by_mtcnn.py --img_path '你所需识别的图片路径'
'''