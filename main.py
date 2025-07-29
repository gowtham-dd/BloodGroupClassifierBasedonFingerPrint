from src.BloodGroupClassifier import logger
from src.BloodGroupClassifier.pipeline.Data_Ingestion_Pipeline import DataIngestionTrainingPipeline
from src.BloodGroupClassifier.pipeline.Data_Preprocessing_Pipeline import DataPreprocessingTrainingPipeline
from src.BloodGroupClassifier.pipeline.Model_Training_Pipeline import ModelTrainingPipeline

# dagshub.init(repo_owner='gowtham-dd', repo_name='Introvert-Vs-Extrovert', mlflow=True)


STAGE_NAME="Data Ingestion stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e




STAGE_NAME="Data Preprocessing stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataPreprocessingTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME="Model Training stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e

