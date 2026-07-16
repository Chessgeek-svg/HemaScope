import pandas as pd
from hemascope.vocab import ATTRIBUTES, VALUE_TO_INDEX, CLASS_TO_INDEX
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2



class MorphologyDataset(Dataset):
    def __init__(self, attributes_filepath, metadata_filepath):
        self.attributes_filepath = attributes_filepath
        self.metadata_filepath = metadata_filepath
            
        df = pd.read_csv(self.attributes_filepath)
        metadata = pd.read_csv(self.metadata_filepath)
        df = df.merge(metadata, on="image_path", how='inner')
        
        for attr in ATTRIBUTES:
            df[attr] = df[attr].map(VALUE_TO_INDEX[attr])
        self.df = df
        
        self.transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(), # PIL -> uint8 tensor, Channel, Height, Width
            v2.ToDtype(torch.float32, scale=True), # uint8 [0,255] -> float [0,1]
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]), # ImageNet hardcoded mean/std, could consider
                                                    # calculating my own from my dataset aggregate
                                                    # to balance out color tint and / or stain quality 
        ])
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 11 attribute targets, in ATTRIBUTES order, as an int64 tensor -> shape (11,)
        attr_targets = torch.tensor(row[ATTRIBUTES].to_numpy(dtype="int64"))

        # class target: string -> canonical int index
        class_target = CLASS_TO_INDEX[row["hemascope_label"]]

        # image tensor still to come
        image_tensor = None
        path = row["image_path"]
        image = Image.open(path).convert("RGB")
        image.resize([224,224])
        image_tensor = self.transform(image)
        
        return image_tensor, attr_targets, class_target


if __name__ == "__main__":
    ds = MorphologyDataset("metadata/attributes.csv", "metadata/metadata.csv")
    print(len(ds))
    print(ds[0])
