import tensorflow as tf
from tensorflow.keras import layers
from .vit import create_vit_model
from tensorflow.keras import regularizers

def create_decoder_branch(inputs, num_filters, num_outputs, name, activation='sigmoid'):
    x = inputs
    for _ in range(4):  # Increased from 3 to 4
        x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
    x = layers.Conv2D(inputs.shape[-1], 1, padding='same', activation='relu')(x)
    x = layers.Add()([x, inputs])  # Add residual connection
    x = layers.Dropout(0.1)(x)  # Added dropout
    outputs = layers.Conv2D(num_outputs, 1, activation=activation, name=name)(x)
    return outputs

def create_he_expert(input_shape, num_classes):
    vit_encoder = create_vit_model(
        input_shape=input_shape,
        patch_size=16,
        num_patches=(input_shape[0] // 16) ** 2,
        projection_dim=384,
        num_transformer_layers=16,  # Increased from 12 to 16
        num_heads=6,
        mlp_head_units=[2048, 1024],
        dropout_rate=0.1,
        num_classes=num_classes['tc_branch'],
    )
    
    inputs = layers.Input(shape=input_shape)
    encoder_output = vit_encoder(inputs)
    
    # Collect skip connections
    skip_outputs = []
    for layer in vit_encoder.layers:
        if isinstance(layer, layers.MultiHeadAttention):
            skip_outputs.append(layer.output)
    
    # Reshape encoder output to 2D
    x = layers.Dense(input_shape[0] * input_shape[1], activation="relu")(encoder_output)
    x = layers.Reshape((input_shape[0], input_shape[1], -1))(x)
    
    # Decoder (adjust to maintain input dimensions)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Use skip connections
    for i, skip in enumerate(reversed(skip_outputs)):
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(64 * (2**i), 3, padding='same', activation='relu')(x)
    
    # Three decoder branches with create_decoder_branch function
    np_branch = create_decoder_branch(x, 64, 1, "np_branch", activation='sigmoid')
    hv_branch = create_decoder_branch(x, 64, 2, "hv_branch", activation='tanh')
    nt_branch = create_decoder_branch(x, 128, num_classes['nt_branch'], "nt_branch", activation='softmax')  # Increased filters to 128
    
    # Tissue classification branch
    tc_branch = layers.GlobalAveragePooling2D()(x)
    tc_branch = layers.Dense(256, activation='relu')(tc_branch)
    tc_branch = layers.Dropout(0.2)(tc_branch)  # Increased dropout to 0.2
    tc_branch = layers.Dense(num_classes['tc_branch'], activation='softmax', name="tc_branch")(tc_branch)
    
    full_model = tf.keras.Model(inputs=inputs, outputs=[np_branch, hv_branch, nt_branch, tc_branch])
    encoder_model = tf.keras.Model(inputs=inputs, outputs=encoder_output)
    
    print("Model output shapes:")
    print(f"NP branch: {np_branch.shape}")
    print(f"HV branch: {hv_branch.shape}")
    print(f"NT branch: {nt_branch.shape}")
    print(f"TC branch: {tc_branch.shape}")
    
    return full_model, encoder_model