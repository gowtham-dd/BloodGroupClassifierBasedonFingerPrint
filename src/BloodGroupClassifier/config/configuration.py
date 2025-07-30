from src.BloodGroupClassifier.constant import *
from src.BloodGroupClassifier.utils.common import read_yaml,create_directories 
from src.BloodGroupClassifier.entity.config_entity import DataIngestionConfig,DataPreprocessingConfig,ModelTrainingConfig,ModelEvaluationConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        params = self.params
    
        return DataPreprocessingConfig(
        data_path=Path("artifacts/data_ingestion/"),  # Hardcoded to match original
        img_height=params.IMG_HEIGHT,
        img_width=params.IMG_WIDTH,
        batch_size=params.BATCH_SIZE,
        validation_split=params.VALIDATION_SPLIT
        )
    


    def get_model_training_config(self) -> ModelTrainingConfig:
        training_config = self.config.model_training
        params = self.params.model_training
    
        create_directories([training_config.root_dir])

        return ModelTrainingConfig(
        root_dir=Path(training_config.root_dir),
        trained_model_path=Path(training_config.trained_model_path),
        input_shape=tuple(params.input_shape),
        epochs=params.epochs,
        batch_size=params.batch_size,
        validation_split=params.validation_split,
        learning_rate=params.learning_rate
        )
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.model_evaluation

        create_directories([config.root_dir])

        return ModelEvaluationConfig(
        root_dir=Path(config.root_dir),
        test_data_path=Path(config.test_data_path),
        model_path=Path(config.model_path),
        metric_file_name=Path(config.metric_file_name),
        mlflow_uri=config.mlflow_uri,
        batch_size=params.batch_size,
        target_metric=params.target_metric,
        all_params=params
        )
