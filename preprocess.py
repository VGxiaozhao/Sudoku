import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from functools import cmp_to_key
from config import Config
import queue


def find_board(src_img: np.ndarray):
    board = (0, 0, 0, 0)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in range(len(contours)):
        epsilon = 0.01 * cv2.arcLength(contours[cnt], True)
        approx = cv2.approxPolyDP(contours[cnt], epsilon, True)

        corners = len(approx)
        if corners == 4:
            x, y, w, h = cv2.boundingRect(contours[cnt])
            if board[2] * board[3] < w * h:
                board = (x, y, w, h)
    cv2.rectangle(result, (board[0], board[1]), (board[0] + board[2], board[1] + board[3]), (255, 0, 0), 1)

    return board


def _delete_line(img: np.ndarray):
    q = queue.Queue()
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x][y]:
                q.put_nowait((x, y))
                break
        if not q.empty():
            break
    while not q.empty():
        x, y = q.get_nowait()
        if img[x][y]:
            img[x][y] = 0
            if x+1 < img.shape[0]:
                q.put_nowait((x+1, y))
            if y+1 < img.shape[1]:
                q.put_nowait((x, y+1))
            if x-1 >= 0:
                q.put_nowait((x-1, y))
            if y-1 >= 0:
                q.put_nowait((x, y-1))


def extract_num(src_img: np.ndarray, board: tuple):
    ret_list = []
    split_img = src_img[board[1]:board[1] + board[3], board[0]:board[0] + board[2]]
    img = cv2.cvtColor(split_img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _delete_line(binary)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cont, hie in zip(contours, hierarchy[0]):
        rect = cv2.boundingRect(cont)
        if max(rect[2], rect[3]) < 10 or min(rect[2], rect[3]) < 3:
            continue
        if rect[2] > rect[3]:
            ret_list.append((rect[0]+board[0], rect[1]+board[1]-(rect[2]-rect[3])//2, rect[2], rect[2]))
        else:
            ret_list.append((rect[0]+board[0]-(rect[3]-rect[2])//2, rect[1]+board[1], rect[3], rect[3]))
    return ret_list


def get_index(board, num_list):
    ids = []
    for item in num_list:
        _x = round((item[0] - board[0]) / (board[2] / 9))
        _y = round((item[1] - board[1]) / (board[3] / 9))
        ids.append((_y, _x))
    return ids


class ImgLocation:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def location(self):
        board = find_board(self.img)
        num_list = extract_num(self.img, board)
        ids = get_index(board, num_list)
        return board, num_list, ids


def main():
    for name in os.listdir('image'):
        loc = ImgLocation(f"image/{name}")
        board, num_list, ids = loc.location()
        for rect in num_list:
            cv2.rectangle(loc.img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)
        plt.imshow(loc.img)
        plt.show()


if __name__ == '__main__':
    main()
