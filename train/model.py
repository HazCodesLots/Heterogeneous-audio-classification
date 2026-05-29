"""
Model factory for DCASE 2026 Task 1.
Routes requests to the appropriate architecture implementation.
"""

# ---------------------------------------------------------------------------
# BST Taxonomy — maps each second-level class index to its top-level parent
# ---------------------------------------------------------------------------
# Top-level codes: m=Music, is=Instrument Sounds, sp=Speech,
#                  fx=Sound Effects, ss=Scene Sounds
BST_CLASSES = [
    "m-sp", "m-si", "m-m",          # Music  (0-2)
    "is-p", "is-s", "is-w",         # Instrument Sounds (3-5)
    "is-k", "is-e",                  # Instrument Sounds cont. (6-7)
    "sp-s", "sp-c", "sp-p",         # Speech (8-10)
    "fx-o", "fx-v", "fx-m",         # Sound Effects (11-13)
    "fx-h", "fx-a", "fx-n",         # Sound Effects cont. (14-16)
    "fx-ex", "fx-el",               # Sound Effects cont. (17-18)
    "ss-n", "ss-i", "ss-u", "ss-s", # Scene Sounds (19-22)
]

# Contiguous label mapping (BST class string → 0-22 integer for model head)
# Use this instead of the raw class_idx column from the CSVs, which is
# non-contiguous (values span 101–504 with gaps).
CLASS_TO_IDX: dict[str, int] = {cls: i for i, cls in enumerate(BST_CLASSES)}

BST_TOP_LEVEL = {
    "m-sp": "m", "m-si": "m", "m-m": "m",
    "is-p": "is", "is-s": "is", "is-w": "is", "is-k": "is", "is-e": "is",
    "sp-s": "sp", "sp-c": "sp", "sp-p": "sp",
    "fx-o": "fx", "fx-v": "fx", "fx-m": "fx",
    "fx-h": "fx", "fx-a": "fx", "fx-n": "fx", "fx-ex": "fx", "fx-el": "fx",
    "ss-n": "ss", "ss-i": "ss", "ss-u": "ss", "ss-s": "ss",
}
# Index → top-level index (for hierarchical metric)
TOP_CODES = ["m", "is", "sp", "fx", "ss"]
SECOND_TO_TOP = [TOP_CODES.index(BST_TOP_LEVEL[c]) for c in BST_CLASSES]

def build_model(model_type: str = "convnext", num_classes: int = 23, **kwargs):
    """
    Factory function to build the requested architecture.
    """
    model_type = model_type.lower()
    
    if model_type == "convnext":
        kwargs.pop("freeze_backbone", None)
        from convnext_se import build_model as build_convnext
        return build_convnext(num_classes=num_classes, **kwargs)
        
    elif model_type == "panns":
        from pann_bst import build_model as build_panns
        return build_panns(num_classes=num_classes, **kwargs)
        
    elif model_type == "clap":
        from clap_bst import build_model as build_clap
        return build_clap(num_classes=num_classes, **kwargs)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'convnext', 'panns', or 'clap'.")
