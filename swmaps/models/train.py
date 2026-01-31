import logging
from pathlib import Path
from typing import Any, List, Tuple, Union

from swmaps.models.base import BaseSalinityModel, BaseSegModel


def train(
    model: Union[BaseSegModel, BaseSalinityModel],
    data_pairs: List[Tuple[Union[str, Path], Union[str, Path]]],
    out_dir: Union[str, Path],
    val_pairs: List[Tuple[Union[str, Path], Union[str, Path]]] = None,
    **kwargs,
) -> Any:
    """
    Generic training orchestrator.

    This function delegates the actual training process to the model's
    internal 'train_model' implementation. This allows different architectures
    (like FarSeg vs. Heuristics) to manage their own unique data loading
    and loss requirements.

    Args:
        model: An instance of a model inheriting from BaseSegModel or BaseSalinityModel.
        data_pairs: List of (image_path, mask_path) for training.
        out_dir: Directory where the model should save its results.
        val_pairs: Optional list of (image_path, mask_path) for validation.
        **kwargs: Arbitrary training parameters (epochs, lr, batch_size)
                  passed from the TOML config.
    """
    logging.info(
        f"Initializing training orchestrator for model: {model.__class__.__name__}"
    )

    if not data_pairs:
        logging.error("No training data pairs provided to the orchestrator.")
        return None

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    return model.train_model(
        data_pairs=data_pairs, out_dir=out_path, val_pairs=val_pairs, **kwargs
    )
