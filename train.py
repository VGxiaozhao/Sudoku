import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from cnn import _model
from config import Config

dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
_size = Config.IMG_SIZE


def get_data():
    cv_data, cv_label = get_cv_data()
    pil_data, pil_label = get_pil_data()
    pil_data.extend(cv_data)
    pil_label.extend(cv_label)
    return pil_data, pil_label


def get_cv_data():
    data, label = [], []
    for num in range(1, 10):
        for thickness in range(2, 6):
            for font in range(8):
                for scale in range(10, 23):
                    img_size = cv2.getTextSize(str(num), font, scale / 10, thickness)
                    for x in range(img_size[0][0]):
                        for y in range(_size, img_size[0][1], -1):
                            if x + img_size[0][0] < 0 or x + img_size[0][0] > _size or \
                                    y - img_size[0][1] < 0 or y - img_size[0][1] > _size:
                                continue
                            img = np.zeros((_size, _size))
                            img = cv2.putText(img, str(num), (x, y), font, scale / 10, 255, thickness)
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


def get_data_loader():
    cv_data, cv_label = get_data()
    cv_data = [item.astype(np.float) / 255 for item in cv_data]
    ax, bx, ay, by = train_test_split(
        cv_data, cv_label, test_size=0.2, shuffle=True)
    ax, ay = torch.tensor(ax, dtype=torch.float32, device=dev), torch.tensor(ay, dtype=torch.int64, device=dev)
    bx, by = torch.tensor(bx, dtype=torch.float32, device=dev), torch.tensor(by, dtype=torch.int64, device=dev)

    train_dl = DataLoader(TensorDataset(ax, ay), batch_size=128, shuffle=True)
    test_dl = DataLoader(TensorDataset(bx, by), batch_size=128)
    print(f"train_dl, len:{train_dl.dataset.__len__()}")
    print(f"test_dl, len:{test_dl.dataset.__len__()}")
    return train_dl, test_dl


def train(model):
    criterion = F.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    train_dl, test_dl = get_data_loader()
    for t in range(3):
        # Forward pass: Compute predicted y by passing x to the model
        for x, y in train_dl:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(t, loss.item())
        model.eval()
        with torch.no_grad():
            right, sum_ = 0, 0
            for x, y in test_dl:
                pred = model(x)
                right += int(sum([label_ == item.argmax() for (label_, item) in zip(y, pred)]))
                sum_ += int(x.shape[0])
            print(f"precision: {right}/{sum_}, {right / sum_} \n")
        torch.save(model.state_dict(), Config.MODEL_DUMP)
        model.train()


if __name__ == '__main__':
    def main():
        train(_model)


    main()
