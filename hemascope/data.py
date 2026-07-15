import pandas as pd
from hemascope.vocab import ATTRIBUTES, VALUE_TO_INDEX, CLASS_TO_INDEX
import torch
from torch.utils.data import Dataset


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
        return image_tensor, attr_targets, class_target


if __name__ == "__main__":
    ds = MorphologyDataset("metadata/attributes.csv", "metadata/metadata.csv")
    print(len(ds))
    print(ds[0])
