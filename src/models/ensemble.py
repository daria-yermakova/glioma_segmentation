import torch
import torch.nn as nn

from src.models.unet import UNet


class Ensemble(nn.Module):
    def __init__(
            self, name, number_of_models, in_channel=4, num_classes=1, num_filters=0
    ):
        super().__init__()

        self.number_of_models = number_of_models
        self.name = name
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.models = nn.ModuleList(
            [
                UNet(
                    name=self.name,
                    in_channel=self.in_channel,
                    out_channels=self.num_classes,
                )
                for _ in range(number_of_models)
            ]
        )

    def forward(self, image):
        logits = [model.forward(image) for model in self.models]
        return logits