from typing import Union

from swmaps.models.farseg import FarSegModel
from swmaps.models.panopticon import PanopticonModel
from swmaps.models.salinity_heuristic import SalinityHeuristicModel

# Import other models as you add them

MODEL_REGISTRY = {
    "farseg": FarSegModel,
    "panopticon": PanopticonModel,
}


def get_model(model_name: str, **kwargs) -> Union[FarSegModel, SalinityHeuristicModel]:
    """
    Factory function to initialize models by name.
    """
    model_name = model_name.lower()

    if model_name == "farseg":
        return FarSegModel(
            num_classes=kwargs.get("num_classes", 16),
            backbone=kwargs.get("backbone", "resnet50"),
        )
    if model_name == "panopticon":
        return PanopticonModel(
            num_classes=kwargs.get("segmentation_num_classes", 256),
            in_channels=kwargs.get("in_channels", 6),
        )

    elif model_name == "salinity_heuristic":
        return SalinityHeuristicModel()

    else:
        raise ValueError(f"Model '{model_name}' is not recognized in model_factory.py")
