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
    
    # Remove flattening and dropout
    # representation = layers.Flatten()(representation)
    # representation = layers.Dropout(0.5)(representation)
    
    # Remove MLP and final dense layer
    # features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout_rate)
    # logits = layers.Dense(num_classes)(features)

    model = tf.keras.Model(inputs=inputs, outputs=representation)
    return model