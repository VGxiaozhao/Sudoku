#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cnn import predict
from config import Config
from preprocess import ImgLocation

_size = Config.IMG_SIZE


class Sudoku:
    def __init__(self, img):
        self.loc = ImgLocation(img)

    def solve(self, index, nums):
        board = np.zeros((9, 9), dtype=np.int8)
        vis_row, vis_col, vis_ret = np.zeros((9, 10), dtype=np.int8), np.zeros((9, 10), dtype=np.int8), np.zeros(
            (9, 10), dtype=np.int8)
        for loc, num in zip(index, nums):
            row, col = loc
            idx = loc[0] // 3 * 3 + loc[1] // 3
            vis_row[row][num] += 1
            vis_col[col][num] += 1
            vis_ret[idx][num] += 1
            board[row][col] = num
        if vis_row.max() > 1:
            return board

        def _dfs(depth):
            if depth >= 81:
                return True

            row, col = depth // 9, depth % 9
            if board[row][col]:
                return _dfs(depth + 1)

            idx = row // 3 * 3 + col // 3
            for i in range(1, 10):
                if not vis_row[row][i] and not vis_col[col][i] and not vis_ret[idx][i]:
                    vis_row[row][i] = vis_col[col][i] = vis_ret[idx][i] = 1
                    board[row][col] = i
                    if _dfs(depth + 1):
                        return True
                    board[row][col] = 0
                    vis_row[row][i] = vis_col[col][i] = vis_ret[idx][i] = 0
            return False

        _dfs(0)
        return board

    def run(self):
        gray = self.loc.gray
        board, points, ids = self.loc.location()
        num, proba = self.classify(gray, points)
        img = self.loc.img

        for point, x in zip(points, num):
            cv2.putText(img, str(int(x)), (point[0], point[1] + point[3]), 0, 2.5, (0, 255, 0), thickness=3)
        ans_board = self.solve(ids, num)
        print(ans_board)
        ox, oy, ow, oh = board
        for x in range(9):
            for y in range(9):
                if (x, y) not in ids and ans_board[x][y]:
                    cv2.putText(img, str(ans_board[x][y]), (ox+y*oh//9, oy+(x+1)*ow//9), 0, 2.5, (255, 0, 0), thickness=3)
        plt.imshow(img)
        plt.show()

    def classify(self, gray, points):
        src_girds, girds = [], []
        for point in points:
            x, y, w, h = point
            src_girds.append(gray[y:y + h, x:x + w])
            tmp = 255 - gray[y:y + h, x:x + w]
            tmp = cv2.resize(tmp, (_size, _size))
            _, tmp = cv2.threshold(tmp, Config.PRE_THRE, 255, 0)
            girds.append(tmp.reshape(-1)/255)
        ret, proba = predict(girds)

        # this code is for check classify error
        # for src, gird, num, p in zip(src_girds, girds, ret, proba):
        #     if num in (5, 3):
        #         print(num, p)
        #         plt.imshow(src)
        #         plt.show()
        #         plt.imshow(gird.reshape(_size, _size))
        #         plt.show()

        return ret, proba


if __name__ == '__main__':
    def main():
        for name in os.listdir('image'):
            print(name)
            a = Sudoku(f'image/{name}')
            a.run()
        a = Sudoku("image/s4.png")
        a.run()

    main()
