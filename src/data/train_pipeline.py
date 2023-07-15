import torch
import numpy as np
import torchvision.transforms as T
import cv2
import numpy as np
import json
import pandas as pd
import segmentation_models_pytorch as smp


class HubMAP_Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int=512, train: bool=True):

        self.df = df
        self.trn = train
        # Explore mean and standard deviation of our data
        self.normalize_image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.image_size = image_size
        if image_size != 512:
            self.resize_image = T.transforms.Resize(image_size)
    
    
    @staticmethod
    def __get_id(path: str) -> str:
        """Extracts image id from path"""
        parts = path.split("/")
        file_name = parts[-1]
        identity = file_name.split(".")[0]
        return identity
    
    
    def __get_image(self, path: str) -> np.array:
        """Gets Image From Path"""
        image = cv2.imread(path)
        return torch.tensor(np.reshape(image, (self.image_size, self.image_size, 3))).to(torch.float32).permute(2, 0, 1)

    
    def __get_mask(self, path: str) -> np.array:
        """Creates Mask From Path"""
        image_id = HubMAP_Dataset.__get_id(path)

        mask = np.zeros((self.image_size, self.image_size, 1), dtype=np.uint8)
        annots = self.df.loc[self.df["id"] == image_id, 'annotations'].iloc[0]
        for annot in annots:
            annot_type = annot['type']
            coordinates = annot['coordinates']
            if annot_type == 'blood_vessel':
                cv2.fillPoly(mask, [np.array(coordinates)], (255,))
        mask = mask/255
        return torch.tensor(mask)
        
    
    def __getitem__(self, index):
        assert index <= len(self.df) and index >= 0, "index needs to be between 0 and length of dataframe"
        assert "path" in self.df.columns, "Path not found in columsn"
        
        row = self.df.iloc[index]
        path = row['path']
        
        img = self.__get_image(path)
        mask = self.__get_mask(path)
        img = self.normalize_image(img)
        
        return img.float(), mask.float()
        

    def __len__(self):
        return len(self.df)