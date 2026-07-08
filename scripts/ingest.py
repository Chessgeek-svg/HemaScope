"""Ingest raw datasets into a single standardized metadata file."""
from pathlib import Path
from typing import Iterator, NamedTuple
import pandas as pd
from sklearn.model_selection import train_test_split


class RawRow(NamedTuple):
    """One image as emitted by an adapter, before taxonomy mapping/splitting."""
    image_path: Path
    source_dataset: str
    original_label: str


def ingest_mll23(root: Path) -> Iterator[RawRow]:
    """Yield one RawRow per image in the MLL23 dataset.

    MLL23 is folder-based: each subdirectory of `root` is a class, and the
    folder name is the original label for every image inside it.
    """
    for folder in root.iterdir():
        if not folder.is_dir() or "MACOSX" in folder.name:
            continue
        for image in folder.glob("*.tif"):
            yield RawRow(image_path=image, source_dataset="mll23", original_label=folder.name)
        


# original MLL23 folder name -> one of the 17 HemaScope classes
MLL23_LABEL_MAP: dict[str, str] = {
    "basophil": "Basophil",
    "eosinophil": "Eosinophil",
    "hairy_cell": "Hairy Cell",
    "lymphocyte": "Lymphocyte",
    "lymphocyte_large_granular": "Lymphocyte",
    "lymphocyte_neoplastic": "Blast",
    "lymphocyte_reactive": "Reactive Lymphocyte",   # NOTE: MLL23 count (33) is unreliable; may drop later
    "metamyelocyte": "Metamyelocyte",
    "monocyte": "Monocyte",
    "myeloblast": "Blast",
    "myelocyte": "Myelocyte",
    "neutrophil_band": "Band Neutrophil",
    "neutrophil_segmented": "Segmented Neutrophil",
    "normoblast": "Erythroblast",
    "plasma_cell": "Plasma Cell",
    "promyelocyte": "Promyelocyte",
    "promyelocyte_atypical": "Promyelocyte",
    "smudge_cell": "Smudge Cell"
}


def assign_splits(rows: list[dict], val_frac: float, test_frac: float) -> None:
    """Add a 'split' key to each row in place. Runs once, on ALL rows."""
    
    labels = [r["hemascope_label"] for r in rows]
    idx = list(range(len(rows)))
    
    trainval_idx, test_idx = train_test_split(
        idx, test_size=test_frac, stratify=labels, random_state=42
    )

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_frac / (1 - test_frac),
        stratify=[labels[i] for i in trainval_idx],
        random_state=42,
    )
    
    for i in train_idx: rows[i]["split"] = "train"
    for i in val_idx: rows[i]["split"] = "val"
    for i in test_idx: rows[i]["split"] = "test"

def main() -> None:
    data_root = Path("data")

    raw: list[RawRow] = []
    raw.extend(ingest_mll23(data_root / "MLL23"))
    #raw.extend(ingest_raabin(...)), etc.

    rows: list[dict] = []
    for r in raw:
        rows.append({
            "image_path": str(r.image_path),
            "source_dataset": r.source_dataset,
            "original_label": r.original_label,
            "hemascope_label": MLL23_LABEL_MAP[r.original_label],  # TODO: per-dataset map lookup
            "split": None,  # filled in by assign_splits
        })

    assign_splits(rows, val_frac=0.15, test_frac=0.15)

    pd.DataFrame(rows).to_csv('metadata/mll23.csv')


if __name__ == "__main__":
    main()
