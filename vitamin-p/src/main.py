# src/main.py

import tensorflow as tf
import numpy as np
from data.preprocessing import preprocess_pannuke_data
from models.expert_he import create_he_expert
from utils.metrics import calculate_metrics
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configurations
    data_config = load_config('configs/data_config.yaml')
    model_config = load_config('configs/model_config.yaml')
    training_config = load_config('configs/training_config.yaml')

    # Preprocess data
    train_dataset, val_dataset, test_dataset, types = preprocess_pannuke_data(
        data_config['he_data_dir'],
        data_config['fold'],
        model_config['batch_size']
    )

    # Print dataset information
    print("Dataset Information:")
    for name, dataset in [("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]:
        print(f"{name} dataset:")
        for images, masks in dataset.take(1):
            print(f"  Image shape: {images.shape}")
            print(f"  Image dtype: {images.dtype}")
            print(f"  Image min and max: {tf.reduce_min(images)}, {tf.reduce_max(images)}")
            print(f"  Mask shape: {masks.shape}")
            print(f"  Mask dtype: {masks.dtype}")
            print(f"  Mask min and max: {tf.reduce_min(masks)}, {tf.reduce_max(masks)}")

    # Create model
    model = create_he_expert(model_config['input_shape'], model_config['num_classes'])
    
    # Print model summary
    model.summary()

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(model_config['learning_rate']),
        loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)]
    )

    # Train model
    history = model.fit(
        train_dataset,
        epochs=training_config['epochs'],
        validation_data=val_dataset,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=training_config['early_stopping_patience']),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=training_config['model_checkpoint_path'],
                save_best_only=True
            )
        ]
    )

    # Evaluate model
    test_loss, test_iou = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test IoU: {test_iou:.4f}")

    # Calculate additional metrics
    y_pred = model.predict(test_dataset)
    y_true = np.concatenate([masks.numpy() for images, masks in test_dataset], axis=0)
    
    # Ensure y_true and y_pred have the same shape
    y_pred = y_pred.reshape(y_true.shape)
    
    metrics = calculate_metrics(y_true, y_pred)
    
    print("Additional Test Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    def check_batches(dataset, num_batches=5):
        for i, (images, masks) in enumerate(dataset.take(num_batches)):
            print(f"Batch {i+1}:")
            print(f"  Image shape: {images.shape}")
            print(f"  Image min and max: {tf.reduce_min(images)}, {tf.reduce_max(images)}")
            print(f"  Mask shape: {masks.shape}")
            print(f"  Mask min and max: {tf.reduce_min(masks)}, {tf.reduce_max(masks)}")

    # In your main function
    check_batches(train_dataset)
    check_batches(val_dataset)
    check_batches(test_dataset)


if __name__ == "__main__":
    main()