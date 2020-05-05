#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
# coding:utf-8
import os

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from config import Config

_size = Config.IMG_SIZE


def get_data():
    cv_data, cv_label = get_cv_data()
    # return cv_data, cv_label
    pil_data, pil_label = get_pil_data()
    # return pil_data, pil_label
    pil_data.extend(cv_data)
    pil_label.extend(cv_label)
    return pil_data, pil_label


def get_cv_data():
    data, label = [], []
    for num in range(1, 10):
        for thickness in range(2, 6):
            for font in range(8):
                # for font in [0, 2, 3, 4, 6]:
                for scale in range(10, 23):
                    # img = cv2.putText(img, "0", (0, 0), i, 2, 255, 2)
                    img_size = cv2.getTextSize(str(num), font, scale / 10, thickness)
                    # if max(img_size[0]) < 21:
                    #     continue
                    # x = (_size - img_size[0][0]) // 2
                    # y = (_size + img_size[0][1]) // 2
                    for x in range(img_size[0][0]):
                        for y in range(_size, img_size[0][1], -1):
                            if x + img_size[0][0] < 0 or x + img_size[0][0] > _size or \
                                    y - img_size[0][1] < 0 or y - img_size[0][1] > _size:
                                continue
                            img = np.zeros((_size, _size))
                            img = cv2.putText(img, str(num), (x, y), font, scale / 10, 255, thickness)
                            # cv2.rectangle(img, (x, y), (x + img_size[0][0], y - img_size[0][1]), 255)
                            # plt.imshow(img)
                            # plt.show()
                            if img.max() == 0:
                                continue
                            data.append(img.reshape(_size ** 2))
                            label.append(num)
    for i in range(1, 10):
        print("count", i, label.count(i))
    return data, label


def get_pil_data():
    data, label = [], []
    for font_type in os.listdir('font'):
        if not font_type.endswith('ttf'):
            continue
        for font_size in range(20, 34):
            font = ImageFont.truetype(os.path.join('font', font_type), size=font_size)
            for num in range(1, 10):
                img = Image.new('L', (_size * 2, _size * 2), 0)
                drawer = ImageDraw.Draw(img)
                drawer.text((0, 0), str(num), font=font, fill=255)
                img = np.array(img)
                cont, hie = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rect = cv2.boundingRect(cont[0])
                if max(rect[2], rect[3]) > _size:
                    continue
                img[img > 0] = 255
                for x in range(_size - rect[3]):
                    for y in range(_size - rect[2]):
                        board = np.zeros((_size, _size))
                        board[x:x + rect[3], y:y + rect[2]] = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                        data.append(board.reshape(-1))
                        label.append(num)
    return data, label

