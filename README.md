# Stock Price Prediction System

This project provides a comprehensive system for predicting stock prices using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models. It fetches historical stock data, preprocesses it, adds technical indicators, trains multiple models, and evaluates their performance.

## Features

-   **Data Loading:** Downloads historical stock data from Yahoo Finance.
-   **Data Exploration:** Provides visualizations of the stock's closing price, trading volume, and return distributions.
-   **Technical Indicators:** Generates a rich set of technical indicators for multivariate analysis, including:
    -   Moving Averages (MA)
    -   Exponential Moving Averages (EMA)
    -   Moving Average Convergence Divergence (MACD)
    -   Relative Strength Index (RSI)
    -   Bollinger Bands
    -   Stochastic Oscillator
-   **Multiple Models:** Implements and compares three different models:
    1.  **RNN (Univariate):** A simple RNN using only the closing price.
    2.  **LSTM (Univariate):** A more advanced LSTM using only the closing price.
    3.  **LSTM (Multivariate):** An LSTM using the closing price plus a range of technical indicators.
-   **Model Training & Evaluation:** Trains the models and evaluates them using standard regression metrics like RMSE, MAE, and R².
-   **Persistent Models:** Automatically saves trained models to disk to avoid retraining on subsequent runs.
-   **Visualization:** Plots training history and compares predicted vs. actual stock prices.

## Requirements

The script requires the following Python libraries:

-   `numpy`
-   `pandas`
-   `matplotlib`
-   `seaborn`
-   `scikit-learn`
-   `yfinance`
-   `pandas-ta`
-   `tensorflow`
-   `joblib`

## Installation

1.  **Clone the repository (or download the script):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv .venv
    ```
    On Windows:
    ```powershell
    .venv\\Scripts\\Activate.ps1
    ```
    On macOS/Linux:
    ```bash
    source .venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn yfinance pandas-ta tensorflow joblib
    ```

## Usage

To run the complete analysis, simply execute the script from your terminal:

```bash
python stock_prediction_system.py
```

### How It Works

1.  **First Run:**
    -   The script will download the historical stock data for the default stock (AAPL).
    -   It will then preprocess the data and create technical indicators.
    -   Three different neural network models will be trained sequentially. This process may take some time.
    -   As each model is trained, it will be saved into a `saved_models/` directory.
    -   Finally, the script will evaluate the models, print a comparison table, and show performance graphs.

2.  **Subsequent Runs:**
    -   The script will detect the saved models in the `saved_models/` directory.
    -   It will load the pre-trained models instead of training them again.
    -   The analysis will run much faster, proceeding directly to evaluation and visualization.

### Customization

You can easily customize the analysis by changing the parameters in the `if __name__ == "__main__":` block at the bottom of `stock_prediction_system.py`.

-   **To change the stock ticker:**
    Modify the `symbol` argument. For example, to analyze Google's stock:
    ```python
    stock_predictor = StockPredictionSystem(symbol='GOOGL', period='5y')
    ```

-   **To change the historical data period:**
    Modify the `period` argument. For example, to use 10 years of data:
    ```python
    stock_predictor = StockPredictionSystem(symbol='AAPL', period='10y')
    ```

### Clearing Saved Models

If you want to force the models to retrain (for example, after changing the model architecture or training parameters), simply delete the `saved_models/` directory. The script will recreate it and save the new models on its next run.
