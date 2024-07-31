# src/models/expert_he.py

import tensorflow as tf
from tensorflow.keras import layers
from .vit import create_vit_model  # Add this import

def create_he_expert(input_shape, num_classes):
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
    
    # Reshape the output to match the input image dimensions
    x = layers.Dense(input_shape[0] * input_shape[1], activation="relu")(x)
    outputs = layers.Reshape((input_shape[0], input_shape[1]))(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model