# FDA1_Personal_Project

# Enhanced Portfolio Analysis Dashboard

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Overview

This project provides a comprehensive, interactive dashboard built with Streamlit for analyzing stock portfolios. It empowers users to fetch historical stock data, configure custom portfolios, analyze performance metrics, visualize trends, perform technical analysis on individual stocks, leverage machine learning models for predictions, and optimize portfolio allocations based on modern portfolio theory.

The dashboard integrates data fetching, financial calculations, machine learning, and interactive visualizations into a single, user-friendly web application.

## Features

The application is divided into several key sections accessible via the sidebar navigation:

1.  **Portfolio Dashboard:**
    * Displays overall portfolio performance metrics (Annual Return, Volatility, Sharpe, Sortino, Calmar, Max Drawdown, Win Rate).
    * Visualizes cumulative portfolio growth against benchmarks (S&P 500) and individual holdings.
    * Shows rolling performance metrics (Return, Volatility, Sharpe) over user-selected windows.
    * Presents a correlation matrix heatmap of stock returns.
    * Lists individual stock metrics and current weights.

2.  **Stock Analysis:**
    * Allows in-depth analysis of a single stock selected from the portfolio.
    * Provides interactive candlestick charts with volume and moving averages.
    * Analyzes return characteristics (daily returns plot, distribution histogram, cumulative return).
    * Calculates and plots key technical indicators (RSI, MACD, Bollinger Bands).

3.  **ML Predictions:**
    * Offers predictive modeling for selected stocks using:
        * **Random Forest / XGBoost:** Predicts N-day future returns based on historical patterns and technical features.
        * **ARIMA:** Forecasts future price movements based on time series analysis.
    * Displays model performance metrics (MSE, MAE, R², AIC, BIC).
    * Visualizes predictions vs. actual data (or forecasts).
    * Shows feature importance for tree-based models.
    * Provides a concrete future return/price prediction (for educational purposes).

4.  **Portfolio Optimization:**
    * Implements portfolio optimization techniques based on Markowitz's Modern Portfolio Theory.
    * Calculates optimal portfolio weights for:
        * Maximum Sharpe Ratio (best risk-adjusted return).
        * Minimum Volatility (lowest risk).
    * Visualizes the Efficient Frontier, showing the optimal risk-return trade-off.
    * Compares the user's current portfolio allocation to the calculated optimal allocations.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

* **Python:** Version 3.7 or higher. You can download it from [python.org](https://www.python.org/downloads/).
* **pip:** Python's package installer (usually comes with Python). You can check by running `pip --version`.
* **Git:** (Optional) If you want to clone the repository directly.

## Installation & Setup

Follow these steps carefully to set up the project environment:

1.  **Get the Code:**
    * **Option A (Git):** Clone the repository (if available) to your local machine:
        ```bash
        git clone <repository_url>
        cd <repository_directory>
        ```
    * **Option B (Manual):** Download the Python script (`app.py` or similar name) and place it in a dedicated project folder on your computer. Open your terminal or command prompt and navigate into that folder:
        ```bash
        cd path/to/your/project/folder
        ```

2.  **Create a Virtual Environment:**
    * It is **highly recommended** to use a virtual environment to isolate project dependencies and avoid conflicts with other Python projects.
    * **On macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * **On Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * You should see `(venv)` prepended to your terminal prompt, indicating the virtual environment is active.

3.  **Create `requirements.txt` File:**
    * Create a file named `requirements.txt` in your project folder.
    * Copy and paste the following lines into this file:
        ```text
        # requirements.txt
        streamlit
        pandas
        numpy
        yfinance
        plotly
        scikit-learn
        xgboost
        statsmodels
        scipy
        ```

4.  **Install Dependencies:**
    * With your virtual environment activated, install all the required Python packages using pip and the `requirements.txt` file:
        ```bash
        pip install -r requirements.txt
        ```
    * This command reads the file and installs the specified versions of each library.

## Running the Application

1.  **Ensure Virtual Environment is Active:** If you closed your terminal, navigate back to the project directory and reactivate the virtual environment (see Step 2 in Installation).
2.  **Run Streamlit:** Execute the following command in your terminal, replacing `app.py` with the actual name of your Python script if different:
    ```bash
    streamlit run app.py
    ```
3.  **Access Dashboard:** Streamlit will start a local web server and automatically open the application in your default web browser. If it doesn't open automatically, the terminal will provide a `Local URL` (usually `http://localhost:8501`) that you can copy and paste into your browser.

## Using the Application

1.  **Sidebar Navigation:** Use the radio buttons under "Navigation" in the left sidebar to switch between the main sections: "Portfolio Dashboard", "Stock Analysis", "ML Predictions", and "Portfolio Optimization".
2.  **Portfolio Configuration:**
    * **Edit Portfolio:** Enter the stock ticker symbols (e.g., `AAPL, MSFT, GOOGL`) you want to analyze, separated by commas. The default is `AAPL, MSFT, GOOGL, AMZN, META`. Press Enter or click outside the text area after editing. The dashboard will update automatically.
    * **Date Range:** Select the start and end dates for the historical data analysis using the date pickers.
    * **Portfolio Weights:** Expand this section to adjust the desired weight (allocation percentage) for each stock in your portfolio using the sliders. The weights are automatically normalized to sum to 100%.
3.  **Interacting with Pages:**
    * **Dashboard:** View overall performance charts and metrics based on your configured portfolio and weights.
    * **Stock Analysis:** Select a specific stock from the dropdown menu to view its detailed price chart, return analysis, and technical indicators.
    * **ML Predictions:** Choose a stock, select a prediction model (Random Forest, XGBoost, ARIMA), configure its parameters (prediction days, test size, ARIMA orders), and click the "Train Model" button to see the results and future predictions.
    * **Portfolio Optimization:** Select an optimization strategy (Max Sharpe, Min Volatility, Efficient Frontier) to see the calculated optimal portfolio metrics and weights compared to your current portfolio. Explore the interactive Efficient Frontier plot.
4.  **Data Loading:** The application caches downloaded data for an hour (`ttl=3600`) to speed up subsequent loads for the same tickers and date ranges. A spinner indicates when data is being fetched or calculations are running.

## Troubleshooting

* **Dependency Errors During Installation:**
    * Ensure your virtual environment is activated correctly.
    * Make sure you have a compatible Python version (3.7+).
    * Try upgrading pip: `pip install --upgrade pip`.
    * If a specific package fails, search online for installation issues related to that package and your operating system.
* **`yfinance` Data Loading Errors:**
    * Check your internet connection.
    * Verify the stock ticker symbols are correct and valid on Yahoo Finance. Some indices or specific stocks might require different symbols (e.g., `^GSPC` for S&P 500).
    * Yahoo Finance sometimes changes its API or experiences temporary issues. Try again later or check the `yfinance` library's GitHub issues page for known problems.
* **ML Model Training Errors:**
    * `ValueError: Not enough data...` or similar: The selected date range might be too short, or the stock might have limited historical data available, especially after calculating features that require lookback periods (like moving averages). Try extending the date range or choosing a different stock.
    * Errors related to `NaN` or infinite values: Although the code attempts to handle `NaNs`, specific data patterns might still cause issues. Check the data quality for the selected stock.
* **Portfolio Optimization Errors:**
    * Optimization requires at least 2 stocks in the portfolio. Add more tickers if needed.
    * The optimization algorithm (`SLSQP`) might fail to converge if the data has issues (e.g., very low variance, perfect correlation). This is less common but possible.
* **General Application Issues:**
    * Try clearing the Streamlit cache: Click the hamburger menu (☰) in the top-right corner of the app and select "Clear cache".
    * Restart the Streamlit application (Ctrl+C in the terminal, then run `streamlit run app.py` again).

## Disclaimer

**This application is for educational and informational purposes only.**

The content, analysis, predictions, and optimizations provided by this dashboard **do not constitute financial advice.** Investing in the stock market involves significant risk, including the potential loss of principal. Past performance is not indicative of future results.

**Always conduct your own thorough research and consult with a qualified financial professional before making any investment decisions.** The creators of this tool are not liable for any investment decisions made based on the information presented here.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find bugs, please feel free to open an issue or submit a pull request (if applicable).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (or add the MIT license text here).
