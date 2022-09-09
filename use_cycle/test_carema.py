from generators import *
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


def build_target_directory(model_name):
    base = "./res/"
    if not os.path.exists(base):
        os.makedirs(base)
    count = 0
    base1 = base + model_name
    while True:
        base = base1 + "_" + str(count)
        if not os.path.exists(base):
            os.makedirs(base)
            break
        else:
            count += 1
    return base


def get_generator(path):
    g = ResnetGenerator(3, 3)
    g.load_state_dict(torch.load(path))
    g.to('cuda' if torch.cuda.is_available() else 'cpu')
    g.eval()
    return g


def is_image(path):
    path = path.split('.')[-1]
    if path == 'jpg' or path == 'png':
        return True
    else:
        return False


def get_transform():
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Resize(768, interpolation=transforms.InterpolationMode.BICUBIC)]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def pre_process(img):
    trans = get_transform()
    img = trans(img)

    return img


def run(model_na):
    model = get_generator('weight/' + model_na)
    source = './sources'
    res = build_target_directory(model_na)
    count = 0
    assert os.path.exists(source), 'please build a directory named sources in this directory'
    print('-----start-----')
    for r, ds, fs in os.walk(source):
        tot = len(fs)
        for f in fs:
            if is_image(f):
                img = Image.open(source + '/' + f)
                img = np.array(img)
                img = pre_process(img).to('cuda' if torch.cuda.is_available() else 'cpu')
                with torch.no_grad():
                    img = (model.forward(img)).cpu().numpy()
                    img = ((np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
                    plt.imsave(res + '/' + str(count) + '.png', img)
                    count += 1
                    print('{} / {} finished'.format(count, tot))
    print('-----All Finished-----')


def carema_run(model_na):
    model = get_generator('weight/' + model_na)
    res = build_target_directory(model_na)
    count = 0
    # 捕获序号为0的摄像头
    cameroCapture = cv2.VideoCapture(0)
    # 创建窗口
    # cv2.namedWindow('window')
    # cv2.setMouseCallback('window',onMouse)
    # 读取帧
    success, frame = cameroCapture.read()
    while success and cv2.waitKey(1) == -1:
        # cv2.imshow('window', frame)
        success, frame = cameroCapture.read()
        img = np.array(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        img = pre_process(img).to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            img = (model.forward(img)).cpu().numpy()
            img = ((np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            cv2.imshow('trans', img)
            cv2.imwrite(res + '/' + str(count) + '.png', img)
            count += 1
            print('{}finished'.format(count))
    # cv2.destroyWindow('window')
    cameroCapture.release()


if __name__ == '__main__':
    model_name = 'nature_photo.pth'  # e.g. nature_photo.pth
    carema = True
    if carema:
        carema_run(model_name)
    else:
        run(model_name)
