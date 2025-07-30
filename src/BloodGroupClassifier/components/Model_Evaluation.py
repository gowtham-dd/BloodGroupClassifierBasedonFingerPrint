import tensorflow as tf
import mlflow
from urllib.parse import urlparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from BloodGroupClassifier.utils.common import save_json
from BloodGroupClassifier import logger
from src.BloodGroupClassifier.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.image_size = (64, 64)

    def _load_validation_generator(self):
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        val_gen = datagen.flow_from_directory(
            self.config.test_data_path,
            target_size=self.image_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        return val_gen

    def _load_model(self):
        return tf.keras.models.load_model(self.config.model_path)

    def evaluate_model(self):
        model = self._load_model()
        val_gen = self._load_validation_generator()

        loss, accuracy = model.evaluate(val_gen)

        metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy)
        }

        save_json(self.config.metric_file_name, metrics)
        logger.info(f"Evaluation metrics: {metrics}")
        return model, metrics

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        try:
            with mlflow.start_run():
                model, metrics = self.evaluate_model()

                # Log hyperparameters
                mlflow.log_params(self.config.all_params)

                # Log metrics
                mlflow.log_metrics(metrics)

                # Log model
                if tracking_url_type_store != "file":
                    mlflow.tensorflow.log_model(
                        model,
                        artifact_path="cnn_model",
                        registered_model_name="Fingerprint_CNN_Model"
                    )
                else:
                    mlflow.log_artifacts(str(self.config.model_path), artifact_path="cnn_model")

                logger.info("Model evaluation and MLflow logging completed.")

        except Exception as e:
            logger.error(f"MLflow logging failed: {e}")
            if mlflow.active_run():
                mlflow.end_run(status="FAILED")
            raise
