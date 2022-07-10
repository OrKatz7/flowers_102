from torch.utils.data import DataLoader, Dataset
import albumentations
import scipy.io
from sklearn import preprocessing
import glob
import pandas as pd
import cv2
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
def parse_data(path="../",seed=42):
    le = preprocessing.LabelEncoder()
    mat = scipy.io.loadmat(f'{path}/imagelabels.mat')
    le.fit(mat['labels'][0])
    labels = le.transform(mat['labels'][0])
    
    mat = scipy.io.loadmat(f'{path}/setid.mat')
    trnid = mat['trnid'][0]
    valid = mat['valid'][0]
    tstid = mat['tstid'][0]
    trnid = list(trnid) + list(valid) + list(tstid)
    trnid,valid,_,_ = train_test_split(trnid,trnid,test_size=0.5, random_state=seed)
    valid,tstid,_,_ = train_test_split(valid,valid,test_size=0.5, random_state=seed)
    
    data = glob.glob(f'{path}/jpg/*')
    data = sorted(data)
    df = pd.DataFrame(data,columns=['image_id'])
    df['class'] = labels
    
    train = df[df.index.isin(trnid)].reset_index(drop=True)
    val = df[df.index.isin(valid)].reset_index(drop=True)
    test = df[df.index.isin(tstid)].reset_index(drop=True)
    train['fold']=1
    val['fold']=0
    folds = pd.concat([train,val]).reset_index(drop=True)
    return folds,test

class CutoutV2(albumentations.DualTransform):
    def __init__(
        self,
        num_holes=8,
        max_h_size=8,
        max_w_size=8,
        fill_value=0,
        always_apply=False,
        p=0.5,
    ):
        super(CutoutV2, self).__init__(always_apply, p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value

    def apply(self, image, fill_value=0, holes=(), **params):
        return albumentations.functional.cutout(image, holes, fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        for _n in range(self.num_holes):
            y = random.randint(0, height)
            x = random.randint(0, width)

            y1 = np.clip(y - self.max_h_size // 2, 0, height)
            y2 = np.clip(y1 + self.max_h_size, 0, height)
            x1 = np.clip(x - self.max_w_size // 2, 0, width)
            x2 = np.clip(x1 + self.max_w_size, 0, width)
            holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("num_holes", "max_h_size", "max_w_size")
    
def get_transforms(data,CFG):
    
    if data == 'train':
        return albumentations.Compose([
            albumentations.Resize(CFG.size, CFG.size),
            albumentations.RandomBrightness(limit=0.2, p=0.75),
            albumentations.RandomContrast(limit=0.2, p=0.75),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.75),
            albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=45, border_mode=0, p=0.75),
            CutoutV2(max_h_size=int(CFG.size * 0.2), max_w_size=int(CFG.size * 0.2), num_holes=1, p=0.75),
            # albumentations.ToGray(p=0.25),
            # ShiftScaleRotate(p=0.5),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    elif data == 'valid':
        return albumentations.Compose([
            albumentations.Resize(CFG.size, CFG.size),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.labels = df['class'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).long()
        return image.transpose(2,0,1), label