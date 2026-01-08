from torch import nn
from torchgeo.models import FarSeg


class FarSegModel(nn.Module):
    """
    Wrapper around TorchGeo's FarSeg for semantic segmentation.
    """

    def __init__(self, backbone="resnet50", num_classes=2, backbone_pretrained=True):
        super().__init__()
        # Initialize the FarSeg model with a pretrained backbone
        self.model = FarSeg(
            backbone=backbone,
            classes=num_classes,
            backbone_pretrained=backbone_pretrained,
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, C, H, W)
        """
        return self.model(x)

    @property
    def device(self):
        return next(self.model.parameters()).device
