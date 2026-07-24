"""Sequential CBM training, stage 2: train only the class head.

Stage 1 (the joint run in train.py) already produced good attribute heads. Here we
FREEZE everything except the class head and retrain just that head on a
class-balanced loader. The class head then learns to separate classes from stable,
accurate concept vectors instead of chasing a moving target and riding the majority
class.
"""

import torch
import torch.nn.functional as F

from hemascope.data import MorphologyDataset
from hemascope.model import Model

ATTR_PATH, METADATA_PATH = "metadata/attributes.csv", "metadata/metadata.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the stage-1 model. Its attribute heads are the good ones we want to keep;
# its class head is the collapsed one we are about to retrain from scratch.
model = Model("resnet50")
model.load_state_dict(
    torch.load("checkpoints/best_model.pt", map_location=device, weights_only=True)
)
model.to(device)

# Freeze the attribute heads so only the class head can learn. (The backbone is
# already frozen inside Model.__init__.) After this, the concept vectors the class
# head sees are fixed and accurate every step.
for param in model.attribute_heads.parameters():
    param.requires_grad = False

# Re-initialize the class head so we start from a clean slate rather than the
# collapsed weights that always predicted eosinophil.
model.class_head.reset_parameters()

trainset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="train")
valset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="val")

# Class-balanced sampling: weight every sample by the inverse of its class size so
# each class appears about equally often.
class_counts = trainset.df["hemascope_label"].value_counts()
sample_weights = [1.0 / class_counts[label] for label in trainset.df["hemascope_label"]]
sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights, num_samples=len(trainset), replacement=True
)
loader = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=sampler)
val_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False)

# Fresh optimizer over the class head, with no weight decay so its weights can
# grow enough to spread the logits instead of being shrunk toward a flat prior.
optimizer = torch.optim.AdamW(model.class_head.parameters(), lr=1e-3, weight_decay=0.0)


def class_accuracy(model, loader, device):
    """Fraction of cells whose predicted class matches the true class."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, _, class_targets in loader:
            images = images.to(device)
            class_targets = class_targets.to(device)
            _, class_logits = model(images)
            correct += (class_logits.argmax(dim=1) == class_targets).sum().item()
            total += class_targets.size(0)
    return correct / total


best_acc = 0.0
for epoch in range(15):
    # backbone stays in eval via Model.train's override; frozen heads won't update.
    model.train()
    for images, _, class_targets in loader:
        images = images.to(device)
        class_targets = class_targets.to(device)

        # Full forward, but only the class head has trainable params, so only it
        # receives gradients. Loss is the class term alone.
        _, class_logits = model(images)
        loss = F.cross_entropy(class_logits, class_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = class_accuracy(model, val_loader, device)
    print(f"Epoch {epoch}: val_class_acc={acc:.3f}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "checkpoints/best_model_seq.pt")
        print(f"  saved new best (val_class_acc={best_acc:.3f})")
