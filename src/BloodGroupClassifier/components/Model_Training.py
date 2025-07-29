import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
from src.BloodGroupClassifier.entity.config_entity import ModelTrainingConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def _build_model(self, num_classes: int) -> tf.keras.Model:
        """Builds the exact model architecture from original code"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=self.config.input_shape),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, train_generator, validation_generator):
        """Trains model with the same parameters as original code"""
        # Check if model already exists
        if Path(self.config.trained_model_path).exists():
            print(f"Model already exists at {self.config.trained_model_path}")
            return tf.keras.models.load_model(self.config.trained_model_path)

        # Build model
        model = self._build_model(num_classes=train_generator.num_classes)
        
        # Train with same parameters as original
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=self.config.epochs,
            steps_per_epoch=train_generator.samples // self.config.batch_size,
            validation_steps=validation_generator.samples // self.config.batch_size
        )

        # Save in Keras format (SavedModel)
        model.save(self.config.trained_model_path)
        print(f"Model saved to {self.config.trained_model_path}")
        
        return history