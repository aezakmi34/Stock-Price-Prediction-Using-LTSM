# Stock Price Prediction using RNN and LSTM
# Comprehensive implementation for Apple Inc. (AAPL) stock prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
import pandas_ta as ta
import warnings
import os
from joblib import dump, load
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class StockPredictionSystem:
    def __init__(self, symbol='AAPL', period='5y'):
        """
        Initialize the Stock Prediction System
        
        Args:
            symbol (str): Stock symbol to predict
            period (str): Time period for historical data
        """
        self.symbol = symbol
        self.period = period
        self.models_dir = 'saved_models'
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {}
        self.history = {}
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
    def load_data(self):
        """Load stock data from Yahoo Finance"""
        print(f"Loading data for {self.symbol}...")
        try:
            self.data = yf.download(self.symbol, period=self.period)
            
            # Handle multi-level column index returned by yfinance
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = self.data.columns.droplevel(1)

            print(f"Data loaded successfully. Shape: {self.data.shape}")
            print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self):
        """Perform basic data exploration"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print("\n=== DATA EXPLORATION ===")
        print(f"Dataset shape: {self.data.shape}")
        print(f"\nDataset info:")
        print(self.data.info())
        print(f"\nDataset description:")
        print(self.data.describe())
        print(f"\nMissing values:")
        print(self.data.isnull().sum())
        
        # Plot stock price evolution
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.data.index, self.data['Close'], label='Close Price', color='blue')
        plt.title(f'{self.symbol} Close Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(self.data.index, self.data['Volume'], label='Volume', color='orange')
        plt.title(f'{self.symbol} Trading Volume Over Time')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.hist(self.data['Close'], bins=50, alpha=0.7, color='green')
        plt.title(f'{self.symbol} Close Price Distribution')
        plt.xlabel('Price ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        returns = self.data['Close'].pct_change().dropna()
        plt.hist(returns, bins=50, alpha=0.7, color='red')
        plt.title(f'{self.symbol} Daily Returns Distribution')
        plt.xlabel('Daily Returns')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_technical_indicators(self):
        """Create technical indicators for multivariate analysis"""
        print("\n=== CREATING TECHNICAL INDICATORS ===")
        
        # Moving Averages
        self.data['MA_5'] = self.data['Close'].rolling(window=5).mean()
        self.data['MA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA_50'] = self.data['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        self.data['EMA_12'] = self.data['Close'].ewm(span=12).mean()
        self.data['EMA_26'] = self.data['Close'].ewm(span=26).mean()
        
        # MACD
        self.data['MACD'] = self.data['EMA_12'] - self.data['EMA_26']
        self.data['MACD_signal'] = self.data['MACD'].ewm(span=9).mean()
        self.data['MACD_histogram'] = self.data['MACD'] - self.data['MACD_signal']
        
        # RSI
        self.data['RSI'] = ta.rsi(self.data['Close'], length=14)
        
        # Bollinger Bands
        bb = ta.bbands(self.data['Close'], length=20)
        if bb is not None and not bb.empty:
            self.data['BB_upper'] = bb['BBU_20_2.0']
            self.data['BB_middle'] = bb['BBM_20_2.0']
            self.data['BB_lower'] = bb['BBL_20_2.0']
            self.data['BB_width'] = self.data['BB_upper'] - self.data['BB_lower']
        else:
            print("Warning: Could not calculate Bollinger Bands. Columns will be filled with NaN.")
            self.data['BB_upper'] = np.nan
            self.data['BB_middle'] = np.nan
            self.data['BB_lower'] = np.nan
            self.data['BB_width'] = np.nan
        
        # Stochastic Oscillator
        stoch = ta.stoch(self.data['High'], self.data['Low'], self.data['Close'])
        if stoch is not None and not stoch.empty:
            self.data['Stoch_K'] = stoch['STOCHk_14_3_3']
            self.data['Stoch_D'] = stoch['STOCHd_14_3_3']
        else:
            print("Warning: Could not calculate Stochastic Oscillator. Columns will be filled with NaN.")
            self.data['Stoch_K'] = np.nan
            self.data['Stoch_D'] = np.nan
        
        # Price-based features
        self.data['Price_change'] = self.data['Close'].pct_change()
        self.data['High_Low_ratio'] = self.data['High'] / self.data['Low']
        self.data['Close_Open_ratio'] = self.data['Close'] / self.data['Open']
        
        # Volume-based features
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_ratio'] = self.data['Volume'] / self.data['Volume_MA']
        
        print("Technical indicators created successfully!")
        print(f"Dataset shape after adding indicators: {self.data.shape}")
        
        # Drop rows with NaN values
        self.data = self.data.dropna()
        print(f"Dataset shape after removing NaN: {self.data.shape}")
    
    def prepare_data_univariate(self, feature='Close', lookback=60, test_size=0.2):
        """
        Prepare data for univariate time series prediction
        
        Args:
            feature (str): Feature to predict
            lookback (int): Number of time steps to look back
            test_size (float): Proportion of data for testing
        """
        print(f"\n=== PREPARING UNIVARIATE DATA ===")
        print(f"Feature: {feature}, Lookback: {lookback}")
        
        # Extract the feature
        data = self.data[feature].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Split the data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Reshape for RNN/LSTM (samples, time steps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Testing data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_data_multivariate(self, lookback=60, test_size=0.2):
        """
        Prepare data for multivariate time series prediction
        
        Args:
            lookback (int): Number of time steps to look back
            test_size (float): Proportion of data for testing
        """
        print(f"\n=== PREPARING MULTIVARIATE DATA ===")
        print(f"Lookback: {lookback}")
        
        # Select features for multivariate analysis
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'MA_5', 'MA_10', 'MA_20', 'RSI', 'MACD', 
                   'BB_width', 'Stoch_K', 'Price_change', 
                   'High_Low_ratio', 'Volume_ratio']
        
        # Ensure all features exist
        available_features = [f for f in features if f in self.data.columns]
        print(f"Using features: {available_features}")
        
        # Prepare the data
        data = self.data[available_features].values
        target = self.data['Close'].values.reshape(-1, 1)
        
        # Scale the features and target separately
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        scaled_features = feature_scaler.fit_transform(data)
        scaled_target = target_scaler.fit_transform(target)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_features)):
            X.append(scaled_features[i-lookback:i])
            y.append(scaled_target[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Split the data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Testing data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_scaler, target_scaler
    
    def build_rnn_model(self, input_shape, units=50, dropout=0.2):
        """Build and compile RNN model"""
        model = Sequential([
            SimpleRNN(units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            SimpleRNN(units, return_sequences=True),
            Dropout(dropout),
            SimpleRNN(units),
            Dropout(dropout),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def build_lstm_model(self, input_shape, units=50, dropout=0.2):
        """Build and compile LSTM model"""
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            LSTM(units, return_sequences=True),
            Dropout(dropout),
            LSTM(units),
            Dropout(dropout),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train_model(self, model, X_train, y_train, X_test, y_test, 
                   model_name, epochs=100, batch_size=32, scalers_to_save=None):
        """Train the model with early stopping and learning rate reduction"""
        print(f"\n=== TRAINING {model_name.upper()} MODEL ===")
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, min_lr=1e-7)
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, lr_reduction],
            verbose=1
        )
        
        # Store model and history
        self.models[model_name] = model
        self.history[model_name] = history
        
        # Save model and scalers
        model_path = os.path.join(self.models_dir, f"{model_name}.keras")
        model.save(model_path)
        print(f"Model saved to {model_path}")

        if scalers_to_save:
            for scaler_name, scaler_obj in scalers_to_save.items():
                scaler_path = os.path.join(self.models_dir, f"{model_name}_{scaler_name}.joblib")
                dump(scaler_obj, scaler_path)
                print(f"Scaler '{scaler_name}' saved to {scaler_path}")
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_test, model_name, scaler=None):
        """Evaluate model performance"""
        print(f"\n=== EVALUATING {model_name.upper()} MODEL ===")
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Inverse transform if scaler is provided
        if scaler is not None:
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        else:
            predictions = predictions.flatten()
            y_actual = y_test
        
        # Calculate metrics
        mse = mean_squared_error(y_actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual, predictions)
        r2 = r2_score(y_actual, predictions)
        
        # Calculate percentage accuracy
        mape = np.mean(np.abs((y_actual - predictions) / y_actual)) * 100
        
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        return {
            'predictions': predictions,
            'actual': y_actual,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def plot_training_history(self, model_names):
        """Plot training history for models"""
        plt.figure(figsize=(15, 10))
        
        for i, model_name in enumerate(model_names):
            if model_name in self.history:
                history = self.history[model_name]
                
                plt.subplot(2, 2, i+1)
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title(f'{model_name} Training History')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, results, model_names, n_points=200):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(15, 10))
        
        for i, model_name in enumerate(model_names):
            if model_name in results:
                result = results[model_name]
                
                plt.subplot(2, 2, i+1)
                
                # Plot last n_points for better visualization
                actual = result['actual'][-n_points:]
                predicted = result['predictions'][-n_points:]
                
                plt.plot(actual, label='Actual', color='blue', alpha=0.7)
                plt.plot(predicted, label='Predicted', color='red', alpha=0.7)
                plt.title(f'{model_name} - Actual vs Predicted\nRMSE: {result["rmse"]:.2f}, R²: {result["r2"]:.3f}')
                plt.xlabel('Time Steps')
                plt.ylabel('Stock Price')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """Run the complete stock prediction analysis"""
        print("="*60)
        print("STOCK PRICE PREDICTION SYSTEM")
        print("="*60)
        
        # Step 1: Load and explore data
        self.load_data()
        if self.data is None:
            return
        
        self.explore_data()
        
        # Step 2: Create technical indicators
        self.create_technical_indicators()
        
        # Step 3: Univariate Analysis (Close price only)
        print("\n" + "="*60)
        print("UNIVARIATE ANALYSIS (CLOSE PRICE)")
        print("="*60)
        
        X_train_uni, X_test_uni, y_train_uni, y_test_uni = self.prepare_data_univariate()
        
        # Build and train RNN model (univariate)
        rnn_uni_path = os.path.join(self.models_dir, 'RNN_Univariate.keras')
        if os.path.exists(rnn_uni_path):
            print("\nLoading pre-trained RNN_Univariate model...")
            rnn_uni = keras.models.load_model(rnn_uni_path)
            self.models['RNN_Univariate'] = rnn_uni
        else:
            rnn_uni = self.build_rnn_model(input_shape=(X_train_uni.shape[1], 1))
            print(f"\nRNN Model Architecture:")
            rnn_uni.summary()
            self.train_model(rnn_uni, X_train_uni, y_train_uni, X_test_uni, y_test_uni, 
                            'RNN_Univariate', epochs=50)
        
        # Build and train LSTM model (univariate)
        lstm_uni_path = os.path.join(self.models_dir, 'LSTM_Univariate.keras')
        if os.path.exists(lstm_uni_path):
            print("\nLoading pre-trained LSTM_Univariate model...")
            lstm_uni = keras.models.load_model(lstm_uni_path)
            self.models['LSTM_Univariate'] = lstm_uni
        else:
            lstm_uni = self.build_lstm_model(input_shape=(X_train_uni.shape[1], 1))
            print(f"\nLSTM Model Architecture:")
            lstm_uni.summary()
            
            self.train_model(lstm_uni, X_train_uni, y_train_uni, X_test_uni, y_test_uni, 
                            'LSTM_Univariate', epochs=50)
        
        # Evaluate univariate models
        results_uni = {}
        results_uni['RNN_Univariate'] = self.evaluate_model(
            self.models['RNN_Univariate'], X_test_uni, y_test_uni, 'RNN_Univariate', self.scaler)
        results_uni['LSTM_Univariate'] = self.evaluate_model(
            self.models['LSTM_Univariate'], X_test_uni, y_test_uni, 'LSTM_Univariate', self.scaler)
        
        # Step 4: Multivariate Analysis
        print("\n" + "="*60)
        print("MULTIVARIATE ANALYSIS (WITH TECHNICAL INDICATORS)")
        print("="*60)
        
        X_train_multi, X_test_multi, y_train_multi, y_test_multi, feature_scaler, target_scaler = \
            self.prepare_data_multivariate()
        
        # Build and train LSTM model (multivariate)
        lstm_multi_path = os.path.join(self.models_dir, 'LSTM_Multivariate.keras')
        scaler_path = os.path.join(self.models_dir, 'LSTM_Multivariate_target_scaler.joblib')
        if os.path.exists(lstm_multi_path) and os.path.exists(scaler_path):
            print("\nLoading pre-trained LSTM_Multivariate model...")
            lstm_multi = keras.models.load_model(lstm_multi_path)
            target_scaler = load(scaler_path)
            self.models['LSTM_Multivariate'] = lstm_multi
        else:
            lstm_multi = self.build_lstm_model(input_shape=(X_train_multi.shape[1], X_train_multi.shape[2]))
            print(f"\nMultivariate LSTM Model Architecture:")
            lstm_multi.summary()
            
            self.train_model(lstm_multi, X_train_multi, y_train_multi, X_test_multi, y_test_multi, 
                            'LSTM_Multivariate', epochs=50, scalers_to_save={'target_scaler': target_scaler})
        
        # Evaluate multivariate model
        results_multi = {}
        results_multi['LSTM_Multivariate'] = self.evaluate_model(
            self.models['LSTM_Multivariate'], X_test_multi, y_test_multi, 'LSTM_Multivariate', target_scaler)
        
        # Step 5: Visualizations
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Plot training histories
        self.plot_training_history(['RNN_Univariate', 'LSTM_Univariate', 'LSTM_Multivariate'])
        
        # Plot predictions
        all_results = {**results_uni, **results_multi}
        self.plot_predictions(all_results, ['RNN_Univariate', 'LSTM_Univariate', 'LSTM_Multivariate'])
        
        # Step 6: Model Comparison
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_df = pd.DataFrame({
            'Model': list(all_results.keys()),
            'RMSE': [all_results[model]['rmse'] for model in all_results.keys()],
            'MAE': [all_results[model]['mae'] for model in all_results.keys()],
            'R²': [all_results[model]['r2'] for model in all_results.keys()],
            'MAPE (%)': [all_results[model]['mape'] for model in all_results.keys()]
        })
        
        print(comparison_df.round(4))
        
        # Find best model
        best_model = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
        print(f"\nBest performing model based on R²: {best_model}")
        
        return all_results, comparison_df

# Example usage
if __name__ == "__main__":
    # Initialize the system
    stock_predictor = StockPredictionSystem(symbol='AAPL', period='5y')
    
    # Run complete analysis
    results, comparison = stock_predictor.run_complete_analysis()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Key Insights:")
    print("1. RNN and LSTM models both capture temporal dependencies in stock prices")
    print("2. LSTM generally performs better than simple RNN due to its ability to handle long-term dependencies")
    print("3. Multivariate models with technical indicators can provide better predictions")
    print("4. The models show the complexity and challenge of stock price prediction")
    print("5. Further improvements can be made with hyperparameter tuning and ensemble methods")
