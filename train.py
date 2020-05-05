import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from font import get_data
from torch.utils.data import TensorDataset, DataLoader
from cnn import _model
from config import Config
import numpy as np

dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


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
    for t in range(15):
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
            # print(pred)
            print(f"{right}/{sum_}, {right / sum_} \n")
        torch.save(model.state_dict(), Config.MODEL_DUMP)
        model.train()


if __name__ == '__main__':
    def main():
        train(_model)

    main()
