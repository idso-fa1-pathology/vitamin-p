import tensorflow as tf
import numpy as np
from data.preprocessing import preprocess_pannuke_data
from models.expert_he import create_he_expert
from utils.metrics import calculate_metrics
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class ShapePrintingCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_data):
        super().__init__()
        self.train_data = train_data

    def on_batch_begin(self, batch, logs=None):
        if batch == 0:
            print("\nChecking shapes on first batch:")
            x, y = next(iter(self.train_data))
            y_pred = self.model(x, training=False)
            for i, output_name in enumerate(['np_branch', 'hv_branch', 'nt_branch', 'tc_branch']):
                print(f"{output_name} - True: {y[output_name].shape}, Pred: {y_pred[i].shape}")

def main():
    # Load configurations
    data_config = load_config('configs/data_config.yaml')
    model_config = load_config('configs/model_config.yaml')
    training_config = load_config('configs/training_config.yaml')

    # Preprocess data
    train_dataset, val_dataset, test_dataset, unique_types = preprocess_pannuke_data(
        data_config['he_data_dir'],
        data_config['fold'],
        model_config['batch_size']
    )

    print(f"Unique tissue types: {unique_types}")

    # Create model
    model = create_he_expert(model_config['input_shape'], len(unique_types))
    model.summary()

    # Compile model with custom losses
    losses = {
        'np_branch': tf.keras.losses.BinaryCrossentropy(from_logits=False),
        'hv_branch': tf.keras.losses.MeanSquaredError(),
        'nt_branch': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        'tc_branch': tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    }
    loss_weights = {
        'np_branch': 1.0,
        'hv_branch': 0.5,
        'nt_branch': 1.0,
        'tc_branch': 1.0
    }
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(model_config['learning_rate']),
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            'np_branch': [tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)],
            'hv_branch': ['mae'],
            'nt_branch': ['accuracy'],
            'tc_branch': ['accuracy']
        }
    )

    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-6
    )

    # Define all callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=training_config['early_stopping_patience']),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=training_config['model_checkpoint_path'],
            save_best_only=True
        ),
        lr_scheduler,
        ShapePrintingCallback(train_dataset)
    ]

    # Train model
    history = model.fit(
        train_dataset,
        epochs=training_config['epochs'],
        validation_data=val_dataset,
        callbacks=callbacks
    )

    # Evaluate model
    test_results = model.evaluate(test_dataset)
    print(f"Test Results: {test_results}")

    # You can add more code here to save the model, plot training history, etc.

if __name__ == "__main__":
    main()