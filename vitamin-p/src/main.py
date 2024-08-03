import tensorflow as tf
import numpy as np
from data.preprocessing import preprocess_pannuke_data
from models.expert_he import create_he_expert
import yaml
import matplotlib.pyplot as plt
import datetime

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

def plot_training_history(history):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, branch in enumerate(['np_branch', 'hv_branch', 'nt_branch', 'tc_branch']):
        ax = axs[i // 2, i % 2]
        ax.plot(history.history[f'{branch}_loss'])
        ax.plot(history.history[f'val_{branch}_loss'])
        ax.set_title(f'{branch} Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Val'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def weighted_bce(class_weights):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weights = tf.reduce_sum(class_weights * y_true, axis=-1)
        return tf.reduce_mean(bce * weights)
    return loss

def weighted_focal_loss(alpha, gamma):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        focal_loss = -alpha * y_true * tf.math.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    return loss

def augment_data(image, np_mask, hv_map, nt_mask, tc_label):
    # Random flip left-right
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        np_mask = tf.image.flip_left_right(np_mask)
        hv_map = tf.image.flip_left_right(hv_map)
        nt_mask = tf.image.flip_left_right(nt_mask)
        hv_map = tf.stack([hv_map[..., 0] * -1, hv_map[..., 1]], axis=-1)

    # Random flip up-down
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        np_mask = tf.image.flip_up_down(np_mask)
        hv_map = tf.image.flip_up_down(hv_map)
        nt_mask = tf.image.flip_up_down(nt_mask)
        hv_map = tf.stack([hv_map[..., 0], hv_map[..., 1] * -1], axis=-1)

    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Ensure image values are still in [0, 1]
    image = tf.clip_by_value(image, 0, 1)

    return image, np_mask, hv_map, nt_mask, tc_label

class WarmUpLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, warmup_batches, init_lr, verbose=0):
        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count * self.init_lr / self.warmup_batches
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
                      'rate to %s.' % (self.batch_count + 1, lr))

class GradientNormLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.gradient_norms = []
        self.learning_rates = []

    def on_train_begin(self, logs=None):
        self.original_train_step = self.model.train_step

        def log_gradient_norm(norm):
            self.gradient_norms.append(float(norm.numpy()))
            return norm

        def log_learning_rate(lr):
            self.learning_rates.append(float(lr.numpy()))
            return lr

        @tf.function
        def train_step_with_gradient_logging(data):
            x, y = data
            with tf.GradientTape() as tape:
                y_pred = self.model(x, training=True)
                loss = self.model.compiled_loss(y, y_pred, regularization_losses=self.model.losses)
            
            # Compute gradients
            gradients = tape.gradient(loss, self.model.trainable_variables)
            
            # Compute gradient norm
            global_norm = tf.linalg.global_norm(gradients)
            
            # Apply gradients
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            # Log gradient norm and learning rate
            tf.py_function(log_gradient_norm, [global_norm], Tout=tf.float32)
            
            if hasattr(self.model.optimizer, 'lr'):
                current_lr = self.model.optimizer.lr
                if callable(current_lr):
                    current_lr = current_lr(self.model.optimizer.iterations)
                tf.py_function(log_learning_rate, [current_lr], Tout=tf.float32)
            elif hasattr(self.model.optimizer, '_decayed_lr'):
                current_lr = self.model.optimizer._decayed_lr(tf.float32)
                tf.py_function(log_learning_rate, [current_lr], Tout=tf.float32)

            # Log to TensorBoard
            tf.summary.scalar('gradient_norm', global_norm, step=self.model.optimizer.iterations)
            tf.summary.scalar('learning_rate', current_lr, step=self.model.optimizer.iterations)
            
            # Update metrics
            self.model.compiled_metrics.update_state(y, y_pred)
            return {m.name: m.result() for m in self.model.metrics}

        self.model.train_step = train_step_with_gradient_logging

    def on_train_end(self, logs=None):
        self.model.train_step = self.original_train_step

    def on_epoch_end(self, epoch, logs=None):
        if self.gradient_norms:
            avg_gradient_norm = sum(self.gradient_norms) / len(self.gradient_norms)
            print(f"\nAverage Gradient Norm for Epoch {epoch + 1}: {avg_gradient_norm:.4f}")
        if self.learning_rates:
            avg_learning_rate = sum(self.learning_rates) / len(self.learning_rates)
            print(f"Average Learning Rate for Epoch {epoch + 1}: {avg_learning_rate:.6f}")
        self.gradient_norms = []
        self.learning_rates = []
       


def main(dry_run=False):
    # Load configurations
    data_config = load_config('configs/data_config.yaml')
    model_config = load_config('configs/model_config.yaml')
    training_config = load_config('configs/training_config.yaml')

    # Preprocess data
    train_dataset, val_dataset, test_dataset, unique_types, class_weight_dict = preprocess_pannuke_data(
        data_config['he_data_dir'],
        data_config['fold'],
        model_config['batch_size'],
        augment_data  # Pass the augmentation function
    )
    print(f"Unique tissue types: {unique_types}")

    # Create model
    model, encoder = create_he_expert(model_config['input_shape'], model_config['num_classes'])
    model.summary()

    # Define learning rate schedule
    initial_learning_rate = model_config['learning_rate']
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.1
    )

    # Define optimizer with gradient clipping
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=0.5)

    # Create weighted loss functions
    np_weights = tf.constant(list(class_weight_dict['np_branch'].values()), dtype=tf.float32)
    nt_weights = tf.constant(list(class_weight_dict['nt_branch'].values()), dtype=tf.float32)
    tc_weights = tf.constant(list(class_weight_dict['tc_branch'].values()), dtype=tf.float32)

    # Compile model with custom losses and class weights
    losses = {
        'np_branch': weighted_bce(np_weights),
        'hv_branch': tf.keras.losses.MeanSquaredError(),
        'nt_branch': weighted_focal_loss(alpha=0.25, gamma=2.0),
        'tc_branch': tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    }
    loss_weights = {
        'np_branch': 1.0,
        'hv_branch': 0.5,
        'nt_branch': 1.0,
        'tc_branch': 1.0
    }

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            'np_branch': [tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)],
            'hv_branch': [tf.keras.metrics.MeanAbsoluteError()],
            'nt_branch': [tf.keras.metrics.CategoricalAccuracy()],
            'tc_branch': [tf.keras.metrics.CategoricalAccuracy()]
        }
    )

    # Define callbacks
    callbacks = [ShapePrintingCallback(train_dataset)]

    if not dry_run:
        # Add additional callbacks for full training
        callbacks.extend([
            tf.keras.callbacks.EarlyStopping(patience=training_config['early_stopping_patience']),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=training_config['model_checkpoint_path'],
                save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-6
            ),
            WarmUpLearningRateScheduler(warmup_batches=1000, init_lr=1e-6, verbose=1),
            tf.keras.callbacks.TensorBoard(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
            GradientNormLogger()  # Use the new GradientNormLogger
        ])

    # Train model with error handling
    try:
        if dry_run:
            print("Starting dry run...")
            history = model.fit(
                train_dataset.take(5),  # Only take 5 batches
                epochs=2,  # Run for 2 epochs
                validation_data=val_dataset.take(2),  # Only take 2 batches for validation
                callbacks=callbacks
            )
            print("Dry run completed successfully!")
        else:
            print("Starting full training...")
            history = model.fit(
                train_dataset,
                epochs=training_config['epochs'],
                validation_data=val_dataset,
                callbacks=callbacks
            )
            print("Full training completed!")


        # Plot training history
        plot_training_history(history)

        # Evaluate model
        test_results = model.evaluate(test_dataset)
        print(f"Test Results: {test_results}")

        # Save the model
        model.save(training_config['final_model_path'])
        
        # Save the encoder separately
        encoder.save(training_config['encoder_model_path'])

        print("Model and encoder saved.")

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

    print("Pipeline test completed.")

if __name__ == "__main__":
    main(dry_run=False)  # Set to False for full training