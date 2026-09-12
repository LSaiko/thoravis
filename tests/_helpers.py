"""Shared test helpers."""

import tempfile

from transformers import ViTConfig, ViTModel


def make_tiny_vit_checkpoint(image_size: int = 224) -> str:
    """Build & save a randomly-initialized, tiny ViT checkpoint locally.

    Lets model tests exercise a real ViT forward pass at the pipeline's
    actual 224x224 resolution without network access or downloading the
    ~350MB real ViT-B/16 backbone.
    """
    config = ViTConfig(
        image_size=image_size,
        patch_size=16,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
    )
    tmp_dir = tempfile.mkdtemp(prefix="thoravis_tiny_vit_")
    ViTModel(config).save_pretrained(tmp_dir)
    return tmp_dir
