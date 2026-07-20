import torch
import torch.nn.functional as F

from hemascope.data import MorphologyDataset
from hemascope.model import Model
from hemascope.vocab import ATTRIBUTES

ATTR_PATH, METADATA_PATH = "metadata/attributes.csv", "metadata/metadata.csv"
ACCUM_STEPS = 4
trainset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="train")
valset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="val")
testset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="test")
model = Model("resnet50")
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0.01
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()  # freeze backbone, set heads to train mode
loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)
scaler = torch.amp.GradScaler('cuda')  # type: ignore


def compute_loss(attr_logits, class_logits, attr_targets, class_targets):
    attr_loss = sum(
        F.cross_entropy(attr_logits[attr], attr_targets[:, i])
        for i, attr in enumerate(ATTRIBUTES)
    )
    class_loss = F.cross_entropy(class_logits, class_targets)
    return attr_loss + class_loss


def evaluate(model, loader, device):
    """Run over `loader` without training; return per-attribute and class accuracy."""
    model.eval()  # disable dropout / BN updates
    attr_correct = {attr: 0 for attr in ATTRIBUTES}
    class_correct = 0
    total = 0
    with torch.no_grad():  # no gradients needed, saves memory
        for images, attr_targets, class_targets in loader:
            images = images.to(device)
            attr_targets = attr_targets.to(device)
            class_targets = class_targets.to(device)

            attr_logits, class_logits = model(images)

            # for each attribute, the prediction is the highest-scoring value index
            for i, attr in enumerate(ATTRIBUTES):
                preds = attr_logits[attr].argmax(dim=1)
                attr_correct[attr] += (preds == attr_targets[:, i]).sum().item()

            class_preds = class_logits.argmax(dim=1)
            class_correct += (class_preds == class_targets).sum().item()
            total += class_targets.size(0)

    attr_acc = {attr: attr_correct[attr] / total for attr in ATTRIBUTES}
    return attr_acc, class_correct / total


for epoch in range(2):
    model.train()  # evaluate() leaves the model in eval; flip heads back each epoch
    running_loss = 0.0
    for i, (images, attr_targets, class_targets) in enumerate(loader):
        images = images.to(device)
        attr_targets = attr_targets.to(device)
        class_targets = class_targets.to(device)

        with torch.amp.autocast('cuda'):  # type: ignore
            attr_logits, class_logits = model(images)
            loss = compute_loss(attr_logits, class_logits, attr_targets, class_targets)
            # scale each grad by 1/ACCUM so the accumulated grads 
            # sum to one normal-size update
            loss /= ACCUM_STEPS 
        scaler.scale(loss).backward()
        if i % ACCUM_STEPS == ACCUM_STEPS - 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * ACCUM_STEPS  # unscale for reporting
        if i % 50 == 0:
            print(
                f"  step {i}/{len(loader)} avg_loss={running_loss / (i + 1):.4f}",
                flush=True,
            )

    # Flush any remaining gradients after the last batch 
    # if it wasn't a multiple of ACCUM_STEPS
    if len(loader) % ACCUM_STEPS != 0:    
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = running_loss / len(loader)
    attr_acc, class_acc = evaluate(model, val_loader, device)
    print(f"Epoch {epoch}: train_loss={avg_loss:.4f}  val_class_acc={class_acc:.3f}")
    for attr, acc in attr_acc.items():
        print(f"    {attr:28s} {acc:.3f}")    
