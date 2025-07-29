


from BloodGroupClassifier.config.configuration import ConfigurationManager
from BloodGroupClassifier.components.Data_Preprocessing import DataPreprocessing
from BloodGroupClassifier import logger
import os
STAGE_NAME = "Data Ingestion stage"


class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        
        try:
            config = ConfigurationManager()

            data_preprocessing_config = config.get_data_preprocessing_config()
            data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
            train_generator, validation_generator = data_preprocessing.get_data_generators()
        except Exception as e:
            raise e




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e