import torch
import torch.nn.functional as F

from hemascope.data import MorphologyDataset
from hemascope.model import Model
from hemascope.vocab import ATTRIBUTES

ATTR_PATH, METADATA_PATH = "metadata/attributes.csv", "metadata/metadata.csv"
trainset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="train")
valset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="val")
testset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="test")
model = Model("resnet50")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
dataiter = iter(loader)
images, attr_targets, class_targets = next(dataiter)
images = images.to(device)
attr_targets = attr_targets.to(device)
class_targets = class_targets.to(device)

def compute_loss(attr_logits, class_logits, attr_targets, class_targets):
    attr_loss = sum(
        F.cross_entropy(attr_logits[attr], attr_targets[:, i])
        for i, attr in enumerate(ATTRIBUTES)
    )
    class_loss = criterion(class_logits, class_targets)
    return attr_loss + class_loss

for i in range(1000):
    optimizer.zero_grad()
    attr_logits, class_logits = model(images)
    loss = compute_loss(attr_logits, class_logits, attr_targets, class_targets)
    loss.backward()
    optimizer.step()
    print(f"Step {i}: loss={loss.item():.4f}")