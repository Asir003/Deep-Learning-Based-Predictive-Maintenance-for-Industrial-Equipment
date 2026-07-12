
from typing import List

import torch
import torch.nn as nn


class PredictiveMaintenanceNet(nn.Module):  

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] | None = None,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        layers: list[nn.Module] = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_size = hidden_size

        # Final output layer with sigmoid for binary classification
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        return self.network(x)
