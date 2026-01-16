from torch import nn


class BaseSegModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        raise NotImplementedError

    def freeze_backbone(self):
        for name, p in self.named_parameters():
            if "backbone" in name:
                p.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device


class BaseSalinityModel(nn.Module):
    """
    Base class for salinity prediction models.

    Supports:
    - Regression (default): predicts continuous salinity values
    - Classification: bin salinity into classes
    """

    def __init__(self, output_dim: int = 1):
        """
        Args:
            output_dim:
                1  -> regression (continuous salinity)
                >1 -> classification (number of salinity classes)
        """
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError

    def freeze_backbone(self):
        for name, p in self.named_parameters():
            if "backbone" in name:
                p.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_regression(self):
        return self.output_dim == 1

    @property
    def is_classification(self):
        return self.output_dim > 1
