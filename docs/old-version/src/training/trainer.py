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
