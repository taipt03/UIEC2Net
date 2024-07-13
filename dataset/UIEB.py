import os
import random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF


class UIEBDataset(data.Dataset):
    def __init__(self, data_path, train_flag=True, pred_flag=False, train_size=256, input_norm=False):
        super(UIEBDataset, self).__init__()
        self.data_path = data_path
        self.train_flag = train_flag if not pred_flag else False
        self.train_size = train_size
        self.pred_flag = pred_flag
        self.input_norm = input_norm
        if self.train_flag:
            self.ann_file = os.path.join(self.data_path, "5K", "train.txt")
        else:
            self.ann_file = os.path.join(self.data_path, "5K", "test.txt")

    def load_annotations(self):
        data_infos = []
        with open(self.ann_file, 'r') as f:
            data_list = f.read().splitlines()
            for data in data_list:
                data_infos.append({
                    "image_path": os.path.join("/kaggle/input/adobe-fivek/raw", data),
                    "gt_path": os.path.join("/kaggle/input/adobe-fivek/c", data),
                    "filename": data,
                })
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        result = self.data_infos[idx]
        data = Image.open(result['image_path']).convert('RGB')
        if not self.pred_flag:
            target = Image.open(result['gt_path']).convert('RGB')
        else:
            target = Image.open(result['image_path']).convert('RGB')
        return data, target, result["filename"]
