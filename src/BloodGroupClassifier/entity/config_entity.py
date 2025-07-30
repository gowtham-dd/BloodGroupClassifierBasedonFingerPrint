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


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path  # This is the folder containing image subdirectories
    model_path: Path
    metric_file_name: Path
    mlflow_uri: str
    batch_size: int
    target_metric: str
    all_params: dict