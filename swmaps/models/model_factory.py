from swmaps.models.farseg import FarSegModel
from swmaps.models.panopticon import PanopticonModel


def get_model(name: str, **kwargs):
    name = name.lower()

    if name == "farseg":
        return FarSegModel(**kwargs)
    if name == "panopticon":
        return PanopticonModel(**kwargs)

    raise ValueError(f"Unknown model: {name}")
