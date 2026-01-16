from torchgeo.models import FarSeg

from swmaps.models.base import BaseSegModel


class FarSegModel(BaseSegModel):
    """
    Wrapper around TorchGeo's FarSeg for semantic segmentation.
    https://arxiv.org/pdf/2011.09766
    """

    def __init__(
        self,
        backbone="resnet50",
        num_classes=2,
        backbone_pretrained=True,
    ):
        super().__init__(num_classes)

        self.model = FarSeg(
            backbone=backbone,
            classes=num_classes,
            backbone_pretrained=backbone_pretrained,
        )

    def forward(self, x):
        return self.model(x)
