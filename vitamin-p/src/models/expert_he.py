import tensorflow as tf
from tensorflow.keras import layers
from .vit import create_vit_model

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
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
