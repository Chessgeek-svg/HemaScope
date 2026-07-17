import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from hemascope import vocab

class Model(nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.feat_dim = self.backbone.num_features
        self.attribute_heads = nn.ModuleDict({
            attr: nn.Linear(self.feat_dim, vocab.num_classes(attr)) #type: ignore
            for attr in vocab.ATTRIBUTES
        })
        self.class_head = nn.Linear(
            in_features=sum(vocab.num_classes(attr) for attr in vocab.ATTRIBUTES), 
            out_features=len(vocab.CLASSES)
        )
        
    def train(self, mode:bool=True):
        super().train(mode)
        self.backbone.eval()
        return self
    

    def forward(self, x):
        feats = self.backbone(x)
        attr_logits = {attr: head(feats) for attr, head in self.attribute_heads.items()}
        probs = [F.softmax(attr_logits[attr], dim=1) for attr in vocab.ATTRIBUTES]
        class_logits = self.class_head(torch.cat(probs, dim=1)) 
        return attr_logits, class_logits

