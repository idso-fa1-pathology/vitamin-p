import tensorflow as tf
import numpy as np
from data.preprocessing import load_and_preprocess_he_data, load_and_preprocess_mif_data, split_data
from training.trainer import VitaminPTrainer
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

    # Load and preprocess data
    he_images, he_masks = load_and_preprocess_he_data(data_config['he_data_dir'])
    mif_images, mif_masks = load_and_preprocess_mif_data(data_config['mif_data_dir'])

    he_data = split_data(he_images, he_masks)
    mif_data = split_data(mif_images, mif_masks)

    # Combine H&E and mIF data for gating network training
    mixed_train_images = np.concatenate([he_data[0][0], mif_data[0][0]])
    mixed_train_labels = np.concatenate([np.zeros(len(he_data[0][0])), np.ones(len(mif_data[0][0]))])

    # Create and train the model
    trainer = VitaminPTrainer(model_config)
    
    print("Training expert models...")
    trainer.train_experts(
        (he_data[0], he_data[1]),  # Train and validation data for H&E
        (mif_data[0], mif_data[1])  # Train and validation data for mIF
    )
    
    print("Training gating network...")
    trainer.train_gating_network((mixed_train_images, mixed_train_labels))
    
    print("Fine-tuning end-to-end...")
    trainer.train_end_to_end((mixed_train_images, mixed_train_labels), epochs=training_config['fine_tune_epochs'])

    # Evaluate the model
    print("Evaluating the model...")
    mixed_test_images = np.concatenate([he_data[2][0], mif_data[2][0]])
    mixed_test_labels = np.concatenate([he_data[2][1], mif_data[2][1]])
    
    predictions = trainer.predict(mixed_test_images)
    metrics = calculate_metrics(mixed_test_labels, predictions)
    
    print("Test Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Save the trained models
    trainer.save_models('models/trained')

if __name__ == "__main__":
    main()
