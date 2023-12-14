import argparse

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


class yolov5_FaceDetection:
    """
    Class implements Yolo5 model to make inferences on a web camera using Opencv2.
    """

    def __init__(self, capture_index, model_name, image=''):
        """
        capture_index表示要使用的摄像头索引，
        model_name表示要加载的YOLOv5模型的名称。
        """
        self.capture_index = capture_index
        if self.capture_index == -1:
            self.image = image
        self.model = self.load_model(model_name)
        self.classes = self.model.names  # {0: 'face'}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        创建一个视频捕获对象，用于逐帧提取视频并进行预测。
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """

        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        如果指定了model_name，则从本地路径加载自定义模型；否则，加载预训练的yolov5s模型。
        :return: Trained Pytorch model.
        """
        if model_name:
            model = torch.hub.load('yolov5', 'custom', path=model_name, force_reload=True, source='local')
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        接受一个帧作为输入，使用YOLOv5模型对该帧进行评分。它返回模型在帧中检测到的对象的标签和坐标。
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]  # (640,640) 图片的大小
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                print((x1, y1), (x2, y2))  # 矩形的左上角坐标为(x1, y1)，右下角坐标为(x2, y2)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def dect_img(self):
        image = self.image
        confidences_list = []
        boxs_list = []
        # 将PIL图像转换为OpenCV图像
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # 调整大小为640x640像素
        image = cv2.resize(image_cv, (640, 640))
        self.model.to(self.device)
        results = self.model(image)  # inference
        # results.show()  # or .show(), .save(), .crop(), .pandas(), etc.
        # print(results)
        labels, cords, preds = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1], results.pred[0]

        # 从张量中提取坐标和置信度
        for i in range(preds.size(0)):
            x1, y1, x2, y2, confidence, _ = preds[i, :]
            confidences_list.append(float(confidence))
            # 转换为整数类型，并计算相对于左上角的坐标
            left = int(x1)
            top = int(y1)
            right = int(x2)
            bottom = int(y2)
            pt1 = (left, top)  # 左上角坐标
            pt2 = (right, bottom)  # 右下角坐标
            boxs_list.append((pt1, pt2))

        image = self.plot_boxes((labels, cords), image)

        cv2.imshow('YOLOv5 Detection', image)
        # 持续显示图像，直到按下任意键
        cv2.waitKey(0)

        # 关闭图像窗口
        cv2.destroyAllWindows()

    def dect_img_one_output(self, img):
        confidences_list = []
        boxs_list = []
        # image = cv2.resize(img, (640, 640))
        self.model.to(self.device)
        results = self.model(img)  # inference
        labels, cords, preds = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1], results.pred[
            0]  # labels: tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.]) (识别出的类别，长度为识别出的数量)

        # 从张量中提取坐标和置信度
        for i in range(preds.size(0)):
            x1, y1, x2, y2, confidence, _ = preds[i, :]
            confidences_list.append(float(confidence))
            # 转换为整数类型，并计算相对于左上角的坐标
            left = int(x1)
            top = int(y1)
            right = int(x2)
            bottom = int(y2)
            pt1 = (left, top)  # 左上角坐标
            pt2 = (right, bottom)  # 右下角坐标
            boxs_list.append((pt1, pt2))

        # 找到最大值的下标
        max_index = confidences_list.index(max(confidences_list))

        # 获取对应下标的值
        max_box = boxs_list[max_index]
        pt1, pt2 = max_box
        left, top = pt1
        right, bottom = pt2

        # 裁剪图像
        cropped_image = img.crop((left, top, right, bottom))
        # 调整大小
        resize = transforms.Resize((224, 224))
        image = resize(cropped_image)

        # 转换为张量
        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(image)

        # 标准化
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        image = normalize(image_tensor)

        # 打印坐标和置信度
        print("检测类别：", self.class_to_label(labels[max_index]))
        print("边界框坐标：", pt1, pt2)
        print("置信度：", max(confidences_list))

        return image, max(confidences_list)

    def dect_frame(self, img):
        confidences_list = []
        boxs_list = []
        # image = cv2.resize(img, (640, 640))
        self.model.to(self.device)
        results = self.model(img)  # inference
        labels, cords, preds = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1], results.pred[
            0]  # labels: tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.]) (识别出的类别，长度为识别出的数量)

        # 从张量中提取坐标和置信度
        for i in range(preds.size(0)):
            x1, y1, x2, y2, confidence, _ = preds[i, :]
            confidences_list.append(float(confidence))
            # 转换为整数类型，并计算相对于左上角的坐标
            left = int(x1)
            top = int(y1)
            right = int(x2)
            bottom = int(y2)
            pt1 = (left, top)  # 左上角坐标
            pt2 = (right, bottom)  # 右下角坐标
            boxs_list.append((pt1, pt2))
            print("检测类别：", self.class_to_label(labels[i]))

        return confidences_list, boxs_list

    def dect_video(self):
        """
                This function is called when class is executed, it runs the loop to read the video frame by frame,
                and write the output into a new file.
                :return: void
                """
        cap = self.get_video_capture()
        assert cap.isOpened()

        while True:

            ret, frame = cap.read()
            assert ret

            frame = cv2.resize(frame, (640, 640))

            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            # print(f"Frames Per Second : {fps}")

            # cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv5 Detection', frame)

            k = cv2.waitKey(1)
            if k % 256 == 27:  # ESC
                print('Esc pressed, closing...')
                break

        cap.release()

    def __call__(self):
        if self.capture_index == -1:
            self.dect_img()
        else:
            self.dect_video()


# image_path = "data/test_img/many_people.jpg"
# image = Image.open(image_path)
# yolov5 = yolov5_FaceDetection(capture_index=-1, model_name='yolov5.pt',
#                               image=image)  # initializing yolov5 for face detection
# yolov5()


def parse_opt(known=False):
    # 创建解析器对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数
    parser.add_argument('--img_path', type=str, help='Your image path')
    # parser.add_argument('--age', type=int, help='Your age')
    parser.add_argument('--cap_index', type=int, help='Your camera index')
    # parser.add_argument('--gender', choices=['male', 'female'], help='Your gender')

    # 解析命令行参数
    args = parser.parse_args()
    return args

def main(opt):
    # image_path = "data/test_img/many_people.jpg"
    image_path = opt.img_path
    image = Image.open(image_path)
    yolov5 = yolov5_FaceDetection(capture_index=opt.cap_index, model_name='yolov5.pt',
                                  image=image)  # initializing yolov5 for face detection
    yolov5()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

'''
命令行输入：
    python face_detection.py --img_path '你所需识别的图片路径'
    python face_detection.py --img_path 'data/test_img/many_people.jpg' --cap_index 0
'''