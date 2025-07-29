## ENTITY
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL:str
    local_data_file:Path
    unzip_dir:Path



@dataclass(frozen=True)
class DataPreprocessingConfig:
    data_path: Path
    img_height: int
    img_width: int
    batch_size: int
    validation_split: float

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    trained_model_path: Path
    input_shape: tuple
    epochs: int
    batch_size: int
    validation_split: float
    learning_rate: float