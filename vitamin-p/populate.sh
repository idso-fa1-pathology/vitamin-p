#!/bin/bash

# Populate src/data/preprocessing.py
cat << EOT > src/data/preprocessing.py
import tensorflow as tf
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

def preprocess_image(image, target_size=(256, 256)):
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image

def load_and_preprocess_he_data(data_dir):
    image_paths = glob.glob(os.path.join(data_dir, "*.png"))
    images = []
    masks = []
    
    for image_path in image_paths:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = preprocess_image(image)
        
        mask_path = image_path.replace("images", "masks")
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = preprocess_image(mask)
        
        images.append(image)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

def load_and_preprocess_mif_data(data_dir):
    # Similar to H&E data loading, but adapted for mIF data format
    pass

def split_data(images, masks, val_split=0.1, test_split=0.1):
    train_images, test_images, train_masks, test_masks = train_test_split(
        images, masks, test_size=test_split, random_state=42
    )
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_images, train_masks, test_size=val_split / (1 - test_split), random_state=42
    )
    return (train_images, train_masks), (val_images, val_masks), (test_images, test_masks)

def main():
    he_data_dir = "data/raw/H&E"
    mif_data_dir = "data/raw/mIF"
    
    he_images, he_masks = load_and_preprocess_he_data(he_data_dir)
    mif_images, mif_masks = load_and_preprocess_mif_data(mif_data_dir)
    
    he_data = split_data(he_images, he_masks)
    mif_data = split_data(mif_images, mif_masks)
    
    # Save processed data
    for data_type, data in zip(["H&E", "mIF"], [he_data, mif_data]):
        for split, (images, masks) in zip(["train", "val", "test"], data):
            np.save(f"data/processed/{data_type}/{split}_images.npy", images)
            np.save(f"data/processed/{data_type}/{split}_masks.npy", masks)

if __name__ == "__main__":
    main()
EOT

# Populate src/models/vit.py
cat << EOT > src/models/vit.py
import tensorflow as tf
from tensorflow.keras import layers

class PatchEmbedding(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEmbedding, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, projection_dim, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential(
            [
                layers.Dense(projection_dim * 2, activation="gelu"),
                layers.Dense(projection_dim),
                layers.Dropout(dropout_rate),
            ]
        )

    def call(self, inputs):
        x = self.norm1(inputs)
        x = self.attn(x, x)
        x = x + inputs
        y = self.norm2(x)
        y = self.mlp(y)
        return x + y

def create_vit_model(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_transformer_layers,
    num_heads,
    mlp_head_units,
    dropout_rate,
    num_classes,
):
    inputs = layers.Input(shape=input_shape)
    patches = layers.Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
    )(inputs)
    patches = layers.Reshape((num_patches, projection_dim))(patches)

    encoded_patches = PatchEmbedding(num_patches, projection_dim)(patches)

    for _ in range(num_transformer_layers):
        encoded_patches = TransformerBlock(
            num_heads, projection_dim, dropout_rate
        )(encoded_patches)

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout_rate)
    logits = layers.Dense(num_classes)(features)
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model
EOT

# Populate src/models/expert_he.py and src/models/expert_mif.py
for expert in he mif; do
cat << EOT > src/models/expert_$expert.py
import tensorflow as tf
from tensorflow.keras import layers
from .vit import create_vit_model

def create_${expert}_expert(input_shape, num_classes):
    vit_model = create_vit_model(
        input_shape=input_shape,
        patch_size=16,
        num_patches=(input_shape[0] // 16) ** 2,
        projection_dim=64,
        num_transformer_layers=8,
        num_heads=4,
        mlp_head_units=[2048, 1024],
        dropout_rate=0.1,
        num_classes=num_classes,
    )
    
    inputs = layers.Input(shape=input_shape)
    x = vit_model(inputs)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
EOT
done

# Populate src/models/gating_network.py
cat << EOT > src/models/gating_network.py
import tensorflow as tf
from tensorflow.keras import layers

def create_gating_network(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
EOT

# Populate src/training/trainer.py
cat << EOT > src/training/trainer.py
import tensorflow as tf
from ..models.expert_he import create_he_expert
from ..models.expert_mif import create_mif_expert
from ..models.gating_network import create_gating_network

class VitaminPTrainer:
    def __init__(self, config):
        self.config = config
        self.he_expert = create_he_expert(config.input_shape, config.num_classes)
        self.mif_expert = create_mif_expert(config.input_shape, config.num_classes)
        self.gating_network = create_gating_network(config.input_shape)
        
        self.he_expert.compile(
            optimizer=tf.keras.optimizers.Adam(config.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        
        self.mif_expert.compile(
            optimizer=tf.keras.optimizers.Adam(config.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        
        self.gating_network.compile(
            optimizer=tf.keras.optimizers.Adam(config.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    def train_experts(self, he_data, mif_data):
        he_train, he_val = he_data
        mif_train, mif_val = mif_data
        
        self.he_expert.fit(
            he_train[0], he_train[1],
            validation_data=he_val,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size
        )
        
        self.mif_expert.fit(
            mif_train[0], mif_train[1],
            validation_data=mif_val,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size
        )

    def train_gating_network(self, mixed_data):
        x_train, y_train = mixed_data
        
        self.gating_network.fit(
            x_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=0.2
        )

    def predict(self, x):
        gate_output = self.gating_network.predict(x)
        he_pred = self.he_expert.predict(x)
        mif_pred = self.mif_expert.predict(x)
        
        # Use the gating network output to weight the predictions
        final_pred = gate_output[:, 0:1] * he_pred + gate_output[:, 1:2] * mif_pred
        return final_pred

    def train_end_to_end(self, mixed_data, epochs):
        x_train, y_train = mixed_data

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Train gating network
            gate_history = self.gating_network.fit(
                x_train, y_train,
                epochs=1,
                batch_size=self.config.batch_size,
                verbose=0
            )
            
            # Get gating predictions
            gate_pred = self.gating_network.predict(x_train)
            
            # Train H&E expert on weighted data
            he_weights = gate_pred[:, 0]
            he_history = self.he_expert.fit(
                x_train, y_train,
                sample_weight=he_weights,
                epochs=1,
                batch_size=self.config.batch_size,
                verbose=0
            )
            
            # Train mIF expert on weighted data
            mif_weights = gate_pred[:, 1]
            mif_history = self.mif_expert.fit(
                x_train, y_train,
                sample_weight=mif_weights,
                epochs=1,
                batch_size=self.config.batch_size,
                verbose=0
            )
            
            # Print metrics
            print(f"Gating Network Loss: {gate_history.history['loss'][0]:.4f}")
            print(f"H&E Expert Loss: {he_history.history['loss'][0]:.4f}")
            print(f"mIF Expert Loss: {mif_history.history['loss'][0]:.4f}")

    def save_models(self, save_path):
        self.he_expert.save(f"{save_path}/he_expert")
        self.mif_expert.save(f"{save_path}/mif_expert")
        self.gating_network.save(f"{save_path}/gating_network")

    def load_models(self, load_path):
        self.he_expert = tf.keras.models.load_model(f"{load_path}/he_expert")
        self.mif_expert = tf.keras.models.load_model(f"{load_path}/mif_expert")
        self.gating_network = tf.keras.models.load_model(f"{load_path}/gating_network")
EOT
# Populate src/utils/metrics.py
cat << EOT > src/utils/metrics.py
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred):
    y_pred_classes = tf.argmax(y_pred, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
EOT

# Populate src/main.py
cat << EOT > src/main.py
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
EOT

# Populate configuration files
cat << EOT > configs/data_config.yaml
he_data_dir: 'data/raw/H&E'
mif_data_dir: 'data/raw/mIF'
processed_data_dir: 'data/processed'
train_split: 0.7
val_split: 0.15
test_split: 0.15
EOT

cat << EOT > configs/model_config.yaml
input_shape: [256, 256, 3]
num_classes: 5  # Adjust based on your specific task
learning_rate: 0.001
batch_size: 32
epochs: 50
EOT

cat << EOT > configs/training_config.yaml
fine_tune_epochs: 20
early_stopping_patience: 5
model_checkpoint_path: 'models/checkpoints'
EOT

# Populate requirements.txt
cat << EOT > requirements.txt
tensorflow==2.6.0
numpy==1.19.5
scikit-learn==0.24.2
opencv-python==4.5.3.56
pyyaml==5.4.1
matplotlib==3.4.3
pandas==1.3.3
EOT

# Populate README.md
cat << EOT > README.md
# Vitamin-P: Vision Transformer Assisted Multi-Modality Expert Network for Pathology

This project implements a multi-expert model for pathology image analysis, combining Vision Transformers with a mixture of experts approach to handle both H&E and mIF image modalities.

## Setup

1. Clone this repository
2. Install the required packages: \`pip install -r requirements.txt\`
3. Place your H&E and mIF datasets in the appropriate directories under \`data/raw/\`
4. Adjust the configuration files in the \`configs/\` directory as needed
5. Run the main script: \`python src/main.py\`

## Project Structure

- \`data/\`: Contains raw and processed data
- \`src/\`: Source code for the project
  - \`data/\`: Data loading and preprocessing
  - \`models/\`: Model architectures
  - \`training/\`: Training pipeline
  - \`utils/\`: Utility functions
- \`configs/\`: Configuration files
- \`notebooks/\`: Jupyter notebooks for exploration and evaluation
- \`tests/\`: Unit tests
- \`models/\`: Saved model checkpoints

## License

[Your chosen license]

## Contact

[Your contact information]
EOT

echo "All files have been populated with initial code and content."
