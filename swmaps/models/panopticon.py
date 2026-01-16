from torchgeo.models import Panopticon

from swmaps.models.base import BaseSegModel


class PanopticonModel(BaseSegModel):
    def __init__(self, num_classes=2):
        super().__init__(num_classes)
        self.model = Panopticon(in_channels=3, classes=num_classes)

    def forward(self, x):
        return self.model(x)
