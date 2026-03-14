"""
Dual-Stream CNN-LSTM Model for Solar Power Prediction
======================================================
Implementation based on: "Solar Power Prediction Using Dual Stream CNN-LSTM Architecture"
Reference: PMC9864442

Architecture:
- Stream 1 (CNN): Extracts spatial features from input sequences
- Stream 2 (LSTM): Captures temporal dependencies
- Attention Mechanism: Refines and weights features
- Dense Layers: Final prediction

Author: [Your Name]
Date: March 2026
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle
import json
from pathlib import Path


class DSCLANet:
    """
    Dual Stream CNN-LSTM with Attention Network (DSCLANet)
    
    Architecture Components:
    1. CNN Stream (3 Conv1D layers) - Spatial feature extraction
    2. LSTM Stream (2 LSTM layers) - Temporal feature extraction  
    3. Attention Mechanism - Feature refinement
    4. Dense Layers - Power prediction
    """
    
    def __init__(self, 
                 input_shape,
                 cnn_filters=[32, 64, 128],
                 cnn_kernels=[5, 3, 1],
                 lstm_units=[100, 100],
                 dense_units=[64, 32, 12],
                 dropout_rate=0.2,
                 learning_rate=0.001):
        """
        Initialize DSCLANet model.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input (lookback_window, n_features), e.g., (2, 18)
        cnn_filters : list
            Number of filters for each Conv1D layer [32, 64, 128]
        cnn_kernels : list
            Kernel sizes for each Conv1D layer [5, 3, 1]
        lstm_units : list
            Number of units for each LSTM layer [100, 100]
        dense_units : list
            Number of units for each Dense layer [64, 32, 12]
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for Adam optimizer
        """
        self.input_shape = input_shape
        self.cnn_filters = cnn_filters
        self.cnn_kernels = cnn_kernels
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_cnn_stream(self, inputs):
        """
        Build CNN stream for spatial feature extraction.
        
        Architecture (as per paper Table 2):
        - Conv1D (32 filters, kernel=5, ReLU)
        - Conv1D (64 filters, kernel=3, ReLU)
        - Conv1D (128 filters, kernel=1, ReLU)
        - Flatten
        
        Returns:
        --------
        cnn_output : tensor
            Flattened spatial features
        """
        # Conv1D Layer 1
        cnn = layers.Conv1D(
            filters=self.cnn_filters[0],
            kernel_size=self.cnn_kernels[0],
            padding='same',
            activation='relu',
            name='cnn_conv1'
        )(inputs)
        cnn = layers.Dropout(self.dropout_rate, name='cnn_dropout1')(cnn)
        
        # Conv1D Layer 2
        cnn = layers.Conv1D(
            filters=self.cnn_filters[1],
            kernel_size=self.cnn_kernels[1],
            padding='same',
            activation='relu',
            name='cnn_conv2'
        )(cnn)
        cnn = layers.Dropout(self.dropout_rate, name='cnn_dropout2')(cnn)
        
        # Conv1D Layer 3
        cnn = layers.Conv1D(
            filters=self.cnn_filters[2],
            kernel_size=self.cnn_kernels[2],
            padding='same',
            activation='relu',
            name='cnn_conv3'
        )(cnn)
        
        # Flatten for concatenation
        cnn_output = layers.Flatten(name='cnn_flatten')(cnn)
        
        return cnn_output
    
    def build_lstm_stream(self, inputs):
        """
        Build LSTM stream for temporal feature extraction.
        
        Architecture (as per paper Table 2):
        - LSTM (100 units, return_sequences=True)
        - LSTM (100 units, return_sequences=False)
        
        Returns:
        --------
        lstm_output : tensor
            Temporal features from LSTM
        """
        # LSTM Layer 1
        lstm = layers.LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            name='lstm_layer1'
        )(inputs)
        lstm = layers.Dropout(self.dropout_rate, name='lstm_dropout1')(lstm)
        
        # LSTM Layer 2
        lstm_output = layers.LSTM(
            units=self.lstm_units[1],
            return_sequences=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            name='lstm_layer2'
        )(lstm)
        lstm_output = layers.Dropout(self.dropout_rate, name='lstm_dropout2')(lstm_output)
        
        return lstm_output
    
    def build_attention_mechanism(self, inputs):
        """
        Build self-attention mechanism for feature refinement.
        
        The attention layer learns to weight features based on their
        importance for prediction, focusing on dominant patterns while
        downweighting less relevant information.
        
        Returns:
        --------
        attention_output : tensor
            Refined features after attention weighting
        """
        # Dense layer for attention scoring
        attention_scores = layers.Dense(
            units=inputs.shape[-1],
            activation='tanh',
            name='attention_scores'
        )(inputs)
        
        # Attention weights (softmax for probability distribution)
        attention_weights = layers.Dense(
            units=inputs.shape[-1],
            activation='softmax',
            name='attention_weights'
        )(attention_scores)
        
        # Apply attention weights
        attention_output = layers.Multiply(name='attention_apply')([inputs, attention_weights])
        
        return attention_output
    
    def build_model(self):
        """
        Build complete Dual-Stream CNN-LSTM model with attention.
        
        Returns:
        --------
        model : keras.Model
            Compiled DSCLANet model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input_layer')
        
        # ==================== STREAM 1: CNN ====================
        print("Building CNN stream...")
        cnn_features = self.build_cnn_stream(inputs)
        
        # ==================== STREAM 2: LSTM ====================
        print("Building LSTM stream...")
        lstm_features = self.build_lstm_stream(inputs)
        
        # ==================== FUSION ====================
        print("Fusing streams...")
        merged = layers.Concatenate(name='fusion_concat')([cnn_features, lstm_features])
        
        # ==================== ATTENTION ====================
        print("Adding attention mechanism...")
        attention_output = self.build_attention_mechanism(merged)
        
        # ==================== DENSE LAYERS ====================
        print("Adding dense layers...")
        dense = layers.Dense(
            units=self.dense_units[0],
            activation='relu',
            name='dense1'
        )(attention_output)
        dense = layers.Dropout(self.dropout_rate, name='dense_dropout1')(dense)
        
        dense = layers.Dense(
            units=self.dense_units[1],
            activation='relu',
            name='dense2'
        )(dense)
        dense = layers.Dropout(self.dropout_rate, name='dense_dropout2')(dense)
        
        dense = layers.Dense(
            units=self.dense_units[2],
            activation='relu',
            name='dense3'
        )(dense)
        
        # ==================== OUTPUT ====================
        output = layers.Dense(
            units=1,
            activation='linear',  # Linear for regression
            name='output'
        )(dense)
        
        # Create model
        model = Model(inputs=inputs, outputs=output, name='DSCLANet')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        
        self.model = model
        print("✓ Model built and compiled successfully!")
        
        return model
    
    def get_model_summary(self):
        """Print detailed model summary"""
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
            return
        
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*70)
        self.model.summary()
        
        # Count parameters by stream
        cnn_params = sum([np.prod(layer.get_weights()[0].shape) 
                         for layer in self.model.layers 
                         if 'cnn' in layer.name and len(layer.get_weights()) > 0])
        
        lstm_params = sum([np.prod(w.shape) 
                          for layer in self.model.layers 
                          if 'lstm' in layer.name 
                          for w in layer.get_weights()])
        
        total_params = self.model.count_params()
        
        print("\n" + "="*70)
        print("PARAMETER BREAKDOWN")
        print("="*70)
        print(f"CNN Stream Parameters    : {cnn_params:,}")
        print(f"LSTM Stream Parameters   : {lstm_params:,}")
        print(f"Other Parameters         : {total_params - cnn_params - lstm_params:,}")
        print(f"Total Trainable Parameters: {total_params:,}")
        print("="*70)
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, batch_size=32, 
              patience=15, save_best_path=None):
        """
        Train the model with callbacks.
        
        Parameters:
        -----------
        X_train : np.array
            Training sequences (n_samples, lookback, n_features)
        y_train : np.array
            Training targets (n_samples,)
        X_val : np.array
            Validation sequences
        y_val : np.array
            Validation targets
        epochs : int
            Maximum number of training epochs
        batch_size : int
            Batch size for training
        patience : int
            Early stopping patience
        save_best_path : str or Path
            Path to save best model weights
        
        Returns:
        --------
        history : History object
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        if save_best_path:
            checkpoint = ModelCheckpoint(
                filepath=save_best_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # Train model
        print("\n" + "="*70)
        print("TRAINING DSCLANet")
        print("="*70)
        print(f"Training samples  : {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Batch size        : {batch_size}")
        print(f"Max epochs        : {epochs}")
        print(f"Early stop patience: {patience}")
        print("="*70 + "\n")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        print("\n✓ Training completed!")
        
        return history
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : np.array
            Input sequences (n_samples, lookback, n_features)
        
        Returns:
        --------
        predictions : np.array
            Predicted values (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data.
        
        Parameters:
        -----------
        X_test : np.array
            Test sequences
        y_test : np.array
            Test targets
        
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        print("\n" + "="*70)
        print("TEST SET EVALUATION")
        print("="*70)
        print(f"MSE  : {mse:.6f}")
        print(f"MAE  : {mae:.6f}")
        print(f"RMSE : {rmse:.6f}")
        print(f"R²   : {r2:.6f}")
        print("="*70)
        
        return metrics
    
    def save_model(self, filepath):
        """Save complete model"""
        if self.model is None:
            raise ValueError("Model not built.")
        self.model.save(filepath)
        print(f"✓ Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load saved model"""
        self.model = keras.models.load_model(filepath)
        print(f"✓ Model loaded from: {filepath}")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_preprocessed_data(data_dir):
    """
    Load preprocessed data from directory.
    
    Parameters:
    -----------
    data_dir : str or Path
        Directory containing preprocessed .npy files
    
    Returns:
    --------
    data_dict : dict
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test
    """
    data_dir = Path(data_dir)
    
    print("Loading preprocessed data...")
    
    data_dict = {
        'X_train': np.load(data_dir / 'X_train.npy'),
        'y_train': np.load(data_dir / 'y_train.npy'),
        'X_val': np.load(data_dir / 'X_val.npy'),
        'y_val': np.load(data_dir / 'y_val.npy'),
        'X_test': np.load(data_dir / 'X_test.npy'),
        'y_test': np.load(data_dir / 'y_test.npy')
    }
    
    print(f"✓ Data loaded:")
    print(f"  X_train: {data_dict['X_train'].shape}")
    print(f"  X_val  : {data_dict['X_val'].shape}")
    print(f"  X_test : {data_dict['X_test'].shape}")
    
    return data_dict


def load_scalers(data_dir):
    """Load saved scalers"""
    scaler_path = Path(data_dir) / 'scalers.pkl'
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    print(f"✓ Scalers loaded from: {scaler_path}")
    return scalers


def inverse_transform_predictions(predictions, scaler):
    """
    Convert normalized predictions back to original scale.
    
    Parameters:
    -----------
    predictions : np.array
        Scaled predictions
    scaler : sklearn scaler
        Target scaler used during preprocessing
    
    Returns:
    --------
    predictions_original : np.array
        Predictions in original scale (Watts)
    """
    predictions_reshaped = predictions.reshape(-1, 1)
    predictions_original = scaler.inverse_transform(predictions_reshaped).flatten()
    return predictions_original


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Example training pipeline using DSCLANet.
    """
    print("\n" + "="*70)
    print(" "*20 + "DSCLANet TRAINING PIPELINE")
    print("="*70)
    
    # ==================== LOAD DATA ====================
    DATA_DIR = Path("/Users/amruthaj/Documents/GitHub/Solar-Prediction-with-ML-DL-Models/PVGIS+NASA_with_DL/preprocessing/preprocessed_data")
    
    # Load preprocessed data
    data = load_preprocessed_data(DATA_DIR)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Load scalers (for inverse transform later)
    scalers = load_scalers(DATA_DIR)
    
    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])  # (lookback, n_features)
    print(f"\nInput shape: {input_shape}")
    
    # ==================== BUILD MODEL ====================
    # Initialize DSCLANet with paper's architecture
    model = DSCLANet(
        input_shape=input_shape,
        cnn_filters=[32, 64, 128],      # As per paper Table 2
        cnn_kernels=[5, 3, 1],          # As per paper Table 2
        lstm_units=[100, 100],          # As per paper Table 2
        dense_units=[64, 32, 12],       # As per paper Table 2
        dropout_rate=0.2,               # For regularization
        learning_rate=0.001             # Adam default
    )
    
    # Build the model
    model.build_model()
    
    # Print summary
    model.get_model_summary()
    
    # ==================== TRAIN MODEL ====================
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32,
        patience=15,
        save_best_path=DATA_DIR / 'best_model.keras'
    )
    
    # ==================== EVALUATE ====================
    metrics = model.evaluate(X_test, y_test)
    
    # ==================== PREDICTIONS ====================
    # Get predictions on test set
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform to original scale
    y_pred_watts = inverse_transform_predictions(y_pred_scaled, scalers['target'])
    y_test_watts = inverse_transform_predictions(y_test, scalers['target'])
    
    # Calculate metrics in original scale
    mse_watts = np.mean((y_test_watts - y_pred_watts) ** 2)
    mae_watts = np.mean(np.abs(y_test_watts - y_pred_watts))
    rmse_watts = np.sqrt(mse_watts)
    
    print("\n" + "="*70)
    print("METRICS IN ORIGINAL SCALE (Watts)")
    print("="*70)
    print(f"MAE  : {mae_watts:.2f} W")
    print(f"RMSE : {rmse_watts:.2f} W")
    print(f"MSE  : {mse_watts:.2f} W²")
    print("="*70)
    
    # ==================== SAVE MODEL ====================
    model.save_model(DATA_DIR / 'final_model.keras')
    
    print("\n✓ Training pipeline completed successfully!")
    print(f"✓ Model and results saved to: {DATA_DIR}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run training pipeline
    main()