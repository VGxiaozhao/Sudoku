import torch


class Config:
    # each img size for cnn classify
    IMG_SIZE = 28
    # for image thread value
    IMG_BINARY_THRE = 200
    # determine how much withe place can a grid contain a num
    WHITE_THRE = 0.90
    # before cnn classify, make a grid become a binary img, this is the threshold
    PRE_THRE = 100
    # torch dump file
    MODEL_DUMP = "./model_state/cnn_state_dict.pth"
    DEV = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
