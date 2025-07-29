from BloodGroupClassifier.components.Data_Preprocessing import DataPreprocessing
from BloodGroupClassifier.components.Model_Training import ModelTrainer

from BloodGroupClassifier.config.configuration import ConfigurationManager
from BloodGroupClassifier import logger
import os
STAGE_NAME = "Data Ingestion stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        
        try:
            config = ConfigurationManager()
    
    # Get data generators (from preprocessing step)
            preprocessing_config = config.get_data_preprocessing_config()
            data_preprocessor = DataPreprocessing(config=preprocessing_config)
            train_gen, val_gen = data_preprocessor.get_data_generators()
    
    # Get model training config
            training_config = config.get_model_training_config()
    
    # Train model
            trainer = ModelTrainer(config=training_config)
            history = trainer.train(train_gen, val_gen)
    
            print("Training completed successfully!")
            print(f"Final model saved to: {training_config.trained_model_path}")

        except Exception as e:
            print(f"Error during model training: {str(e)}")
            raise e




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e