import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from BloodGroupClassifier import logger
from src.BloodGroupClassifier.entity.config_entity import DataPreprocessingConfig


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def get_data_generators(self):
        try:
            # IDENTICAL to original preprocessing
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=self.config.validation_split
            )

            train_generator = train_datagen.flow_from_directory(
                str(self.config.data_path),  # Convert Path to string
                target_size=(self.config.img_height, self.config.img_width),
                batch_size=self.config.batch_size,
                class_mode='categorical',
                subset='training'
            )

            validation_generator = train_datagen.flow_from_directory(
                str(self.config.data_path),
                target_size=(self.config.img_height, self.config.img_width),
                batch_size=self.config.batch_size,
                class_mode='categorical',
                subset='validation'
            )

            logger.info("Created generators with:")
            logger.info(f"  - Image size: {self.config.img_height}x{self.config.img_width}")
            logger.info(f"  - Batch size: {self.config.batch_size}")
            logger.info(f"  - Validation split: {self.config.validation_split}")

            return train_generator, validation_generator

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise e