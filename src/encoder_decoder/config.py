from dataclasses import dataclass


@dataclass
class Config:
    seed = 10
    batch_size: int = 4
    epochs = 3
    max_length: int = 256

    num_hidden_layers: int = 2

    learning_rate: float = 1e-8
    weight_decay: float = 1e-5

    input_path: str = "data"
    output_path: str = "models"
