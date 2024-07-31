import tensorflow as tf
from tensorflow.keras import layers
from .vit import create_vit_model
from tensorflow.keras import regularizers

def create_decoder_branch(inputs, num_filters, num_outputs, name):
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    outputs = layers.Conv2D(num_outputs, 1, activation='sigmoid', name=name)(x)
    return outputs

def create_he_expert(input_shape, num_classes):
    vit_encoder = create_vit_model(
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
    encoder_output = vit_encoder(inputs)
    
    # Reshape encoder output to 2D
    x = layers.Dense(input_shape[0] * input_shape[1], activation="relu")(encoder_output)
    x = layers.Reshape((input_shape[0], input_shape[1], -1))(x)
    
    # Decoder (adjust to maintain input dimensions)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    
    # Three decoder branches
    np_branch = layers.Conv2D(1, 1, activation='sigmoid', name="np_branch")(x)
    hv_branch = layers.Conv2D(2, 1, activation='tanh', name="hv_branch")(x)
    nt_branch = layers.Conv2D(6, 1, activation='softmax', name="nt_branch")(x)
    
    # Tissue classification branch
    tc_branch = layers.GlobalAveragePooling2D()(x)
    tc_branch = layers.Dense(num_classes, activation='softmax', name="tc_branch")(tc_branch)
    
    model = tf.keras.Model(inputs=inputs, outputs=[np_branch, hv_branch, nt_branch, tc_branch])

    print("Model output shapes:")
    print(f"NP branch: {np_branch.shape}")
    print(f"HV branch: {hv_branch.shape}")
    print(f"NT branch: {nt_branch.shape}")
    print(f"TC branch: {tc_branch.shape}")

    return model