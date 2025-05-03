import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize

# Set page configuration
st.set_page_config(
    page_title="Enhanced Portfolio Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Application title and description
st.title("Enhanced Portfolio Analysis Dashboard")
st.markdown("""
This dashboard allows you to analyze a portfolio of stocks, optimize using efficient frontier, 
compare returns, and predict future movements with machine learning models.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Portfolio Dashboard", "Stock Analysis", "ML Predictions", "Portfolio Optimization"])

# Sidebar for portfolio configuration
st.sidebar.title("Portfolio Configuration")

# Default stocks
default_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
with st.sidebar.expander("Edit Portfolio", expanded=True):
    portfolio_input = st.text_area("Enter stock tickers (comma-separated):", ", ".join(default_stocks))
    portfolio_stocks = [ticker.strip() for ticker in portfolio_input.split(",")]

# Date range selection
st.sidebar.title("Date Range")
today = datetime.date.today()
start_date = st.sidebar.date_input("Start Date", today - timedelta(days=365*2))
end_date = st.sidebar.date_input("End Date", today)

# Weights configuration
with st.sidebar.expander("Portfolio Weights", expanded=False):
    st.markdown("Adjust the weights of each stock in your portfolio")
    weights = {}
    # Default: equal weights
    default_weight = 1.0 / len(portfolio_stocks)
    
    for ticker in portfolio_stocks:
        weights[ticker] = st.slider(f"{ticker} weight", 0.0, 1.0, default_weight, 0.01)
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        for ticker in weights:
            weights[ticker] /= total_weight
    
    # Display normalized weights
    st.write("Normalized weights:")
    for ticker, weight in weights.items():
        st.write(f"{ticker}: {weight:.2%}")

# Function to load stock data
@st.cache_data(ttl=3600)
def load_stock_data(ticker_symbol, start_date, end_date):
    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.warning(f"Error loading data for {ticker_symbol}: {e}")
        return None

# Function to load portfolio data
@st.cache_data(ttl=3600)
def load_portfolio_data(tickers, start_date, end_date):
    portfolio_data = {}
    all_data_empty = True
    
    for ticker in tickers:
        data = load_stock_data(ticker, start_date, end_date)
        if data is not None and not data.empty:
            portfolio_data[ticker] = data
            all_data_empty = False
    
    # Create portfolio returns DataFrame
    if all_data_empty:
        return None, None, None
    
    # Extract close prices and calculate returns
    portfolio_closes = pd.DataFrame()
    portfolio_returns = pd.DataFrame()
    
    for ticker, data in portfolio_data.items():
        portfolio_closes[ticker] = data['Close']
        portfolio_returns[ticker] = data['Close'].pct_change()
    
    # Drop first row (NaN values from pct_change)
    portfolio_returns = portfolio_returns.dropna(how='all')
    
    return portfolio_data, portfolio_returns, portfolio_closes

# Calculate portfolio return based on weights
def calculate_portfolio_return(returns_df, weights_dict):
    weighted_returns = pd.DataFrame()
    
    # Create a copy with only the tickers that have weights
    valid_tickers = [ticker for ticker in weights_dict.keys() if ticker in returns_df.columns]
    
    # Calculate weighted return for each ticker
    for ticker in valid_tickers:
        weighted_returns[ticker] = returns_df[ticker] * weights_dict[ticker]
    
    # Sum across columns to get portfolio return
    portfolio_return = weighted_returns.sum(axis=1)
    
    return portfolio_return

# Calculate portfolio performance metrics
def calculate_portfolio_metrics(returns_series):
    """Calculate key portfolio performance metrics"""
    try:
        # Daily metrics
        daily_return = returns_series.mean()
        daily_volatility = returns_series.std()
        
        # Annualized metrics (assuming 252 trading days)
        annual_return = (1 + daily_return) ** 252 - 1
        annual_volatility = daily_volatility * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns_series).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Win rate (percentage of positive days)
        win_rate = (returns_series > 0).mean()
        
        # Calculate Sortino ratio (downside risk only)
        negative_returns = returns_series[returns_series < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate Calmar ratio (return / max drawdown)
        calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'Daily Return': daily_return,
            'Daily Volatility': daily_volatility,
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate
        }
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return {
            'Daily Return': 0,
            'Daily Volatility': 0,
            'Annual Return': 0,
            'Annual Volatility': 0,
            'Sharpe Ratio': 0,
            'Sortino Ratio': 0,
            'Calmar Ratio': 0,
            'Max Drawdown': 0,
            'Win Rate': 0
        }

# Machine Learning Functions
def create_features(stock_data, ticker, window_sizes=[5, 10, 20, 50], target_days=5):
    """Create features for machine learning models"""
    df = stock_data[ticker].copy()
    
    # Basic price and volume features
    df['return'] = df['Close'].pct_change()
    df['volume_change'] = df['Volume'].pct_change()
    df['high_low_diff'] = (df['High'] - df['Low']) / df['Close']
    df['close_open_diff'] = (df['Close'] - df['Open']) / df['Open']
    
    # Moving averages - with error handling
    for window in window_sizes:
        df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
        # Handle division by zero or NaN values
        ma_column = df[f'ma_{window}']
        df[f'ma_diff_{window}'] = np.where(ma_column > 0, 
                                   (df['Close'].values / ma_column.values) - 1, 
                                   0)
        df[f'vol_ma_{window}'] = df['Volume'].rolling(window=window).mean()
        df[f'return_ma_{window}'] = df['return'].rolling(window=window).mean()
        df[f'return_std_{window}'] = df['return'].rolling(window=window).std()
    
    # Target variable: future return over target_days
    df[f'target_{target_days}d'] = df['Close'].pct_change(periods=target_days).shift(-target_days)
    
    # Calculate RSI (Relative Strength Index) with error handling
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # Avoid division by zero
    loss = loss.replace(0, np.nan)
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Fill any remaining NaN values with 0 to avoid ML model errors
    df = df.fillna(0)
    # Add at the end of create_features function

    # Print the shape of the dataframe and check for NaN values
    print(f"DataFrame shape: {df.shape}")
    # Make sure there are no NaN values
    df = df.fillna(0)
    return df
    
    return df

def train_ml_model(data, model_type='random_forest', target_days=5, test_size=0.2):
    """Train a machine learning model for stock price prediction"""
    # Define target and features
    target_col = f'target_{target_days}d'
    
    try:
        # Drop columns that shouldn't be features
        # Print the shape of the dataframe for debugging
        print(f"Original data shape: {data.shape}")

        # Drop columns that shouldn't be features
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', target_col]
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        # Check for empty feature columns
        if len(feature_cols) == 0:
            raise ValueError("No feature columns available after exclusions")

        X = data[feature_cols]
        y = data[target_col]

        # Handle NaN values in both features and target
        X = X.fillna(0)
        y = y.fillna(0)

        # Print shapes before conversion for debugging
        print(f"X shape before conversion: {X.shape}")
        print(f"y shape before conversion: {y.shape}")

        # Convert to numpy arrays with explicit 1D/2D handling
        X_np = X.values
        if len(X_np.shape) == 1:
            X_np = X_np.reshape(-1, 1)  # Make sure X is 2D

        y_np = y.values
        if len(y_np.shape) > 1:
            y_np = y_np.flatten()  # Make sure y is 1D

        # Print shapes after conversion for debugging
        print(f"X_np shape after conversion: {X_np.shape}")
        print(f"y_np shape after conversion: {y_np.shape}")

        # If y_np is a 2D array when it should be 1D, flatten it
        if len(y_np.shape) > 1 and y_np.shape[1] == 1:
            y_np = y_np.flatten()
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_np)
        
        # Use the last split for final testing
        splits = list(tscv.split(X_scaled))
        if len(splits) > 0:
            train_index, test_index = splits[-1]
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y_np[train_index], y_np[test_index]
            
            # Train model based on selected type
            if model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                model.fit(X_train, y_train)
            elif model_type == 'xgboost':
                model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
                model.fit(X_train, y_train)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance (for interpretability)
            if model_type in ['random_forest', 'xgboost']:
                importance = model.feature_importances_
                features_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
            else:
                features_df = None
            
            # Get the dates for the test set
            test_dates = data.index[test_index]
            
            # Create a predictions DataFrame
            predictions_df = pd.DataFrame({
                'Date': test_dates,
                'Actual': y_test,
                'Predicted': y_pred
            })
            
            return model, predictions_df, {'MSE': mse, 'MAE': mae, 'R2': r2}, features_df, scaler
        else:
            raise ValueError("Not enough data for time series split")
    except Exception as e:
        raise Exception(f"Error in training ML model: {str(e)}")


def train_arima_model(data, ticker, p=5, d=1, q=0, forecast_days=30):
    """Train ARIMA model for time series forecasting"""
    try:
        # Get closing prices
        prices = data[ticker]['Close']
        
        # Fit ARIMA model
        model = ARIMA(prices, order=(p, d, q))
        results = model.fit()
        
        # Make forecast
        forecast = results.forecast(steps=forecast_days)
        
        # Create forecast DataFrame
        last_date = prices.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast
        })
        
        # Calculate model metrics
        aic = results.aic
        bic = results.bic
        
        return results, forecast_df, {'AIC': aic, 'BIC': bic}
    except Exception as e:
        st.error(f"Error in ARIMA model: {e}")
        return None, None, None

def predict_future_returns(model, scaler, latest_data, target_days=5):
    """Predict future returns using the trained model"""
    # Preprocess the latest data (same as in training)
    X = latest_data
    X_scaled = scaler.transform(X)
    
    # Make prediction
    predicted_return = model.predict(X_scaled)[0]
    
    return predicted_return

# Portfolio Optimization Functions
def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
    """Calculate portfolio performance (return, volatility, Sharpe ratio)"""
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = returns / std if std > 0 else 0
    return returns, std, sharpe_ratio

def negative_sharpe_ratio(weights, mean_returns, cov_matrix):
    """Objective function to minimize (negative Sharpe ratio)"""
    return -portfolio_annualized_performance(weights, mean_returns, cov_matrix)[2]

def max_sharpe_ratio(mean_returns, cov_matrix, tickers, risk_free_rate=0):
    """Find the portfolio weights that maximize the Sharpe ratio"""
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    opt_results = minimize(negative_sharpe_ratio, num_assets * [1./num_assets,], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Create dictionary of weights
    weights_dict = {ticker: weight for ticker, weight in zip(tickers, opt_results['x'])}
    
    # Calculate performance metrics
    returns, std, sharpe = portfolio_annualized_performance(opt_results['x'], mean_returns, cov_matrix)
    
    return weights_dict, returns, std, sharpe

def min_volatility(mean_returns, cov_matrix, tickers):
    """Find the portfolio weights that minimize volatility"""
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    
    def portfolio_volatility(weights, mean_returns, cov_matrix):
        return portfolio_annualized_performance(weights, mean_returns, cov_matrix)[1]
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    opt_results = minimize(portfolio_volatility, num_assets * [1./num_assets,], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Create dictionary of weights
    weights_dict = {ticker: weight for ticker, weight in zip(tickers, opt_results['x'])}
    
    # Calculate performance metrics
    returns, std, sharpe = portfolio_annualized_performance(opt_results['x'], mean_returns, cov_matrix)
    
    return weights_dict, returns, std, sharpe

def efficient_frontier(mean_returns, cov_matrix, tickers, returns_range=None):
    """Calculate the efficient frontier"""
    num_assets = len(mean_returns)
    
    # Optimal portfolio for minimum volatility
    min_vol_weights, min_vol_return, min_vol_std, min_vol_sharpe = min_volatility(mean_returns, cov_matrix, tickers)
    
    # Optimal portfolio for maximum Sharpe ratio
    max_sharpe_weights, max_sharpe_return, max_sharpe_std, max_sharpe_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, tickers)
    
    if returns_range is None:
        # Create a range of target returns between min and max return
        min_return = min_vol_return - 0.05
        max_return = max_sharpe_return + 0.05
        returns_range = np.linspace(min_return, max_return, 50)
    
    efficient_portfolios = []
    
    # For each target return, find the portfolio with minimum variance
    for target_return in returns_range:
        args = (mean_returns, cov_matrix)
        
        def portfolio_volatility(weights, mean_returns, cov_matrix):
            return portfolio_annualized_performance(weights, mean_returns, cov_matrix)[1]
        
        # Constraints: sum of weights = 1, portfolio return = target
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: portfolio_annualized_performance(x, mean_returns, cov_matrix)[0] - target_return}
        )
        
        bounds = tuple((0, 1) for asset in range(num_assets))
        
        # Initial guess: equal weights
        initial_weights = num_assets * [1./num_assets,]
        
        try:
            result = minimize(portfolio_volatility, initial_weights, args=args,
                           method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result['success']:
                returns, std, sharpe = portfolio_annualized_performance(result['x'], mean_returns, cov_matrix)
                efficient_portfolios.append({'Return': returns, 'Volatility': std, 'Sharpe': sharpe})
        except:
            # Skip if optimization fails
            pass
    
    # Convert to DataFrame
    efficient_df = pd.DataFrame(efficient_portfolios)
    
    # Add optimal portfolios
    min_vol_point = {'Return': min_vol_return, 'Volatility': min_vol_std, 'Sharpe': min_vol_sharpe, 'Portfolio': 'Min Volatility'}
    max_sharpe_point = {'Return': max_sharpe_return, 'Volatility': max_sharpe_std, 'Sharpe': max_sharpe_sharpe, 'Portfolio': 'Max Sharpe'}
    
    # Return the efficient frontier and optimal portfolios
    return efficient_df, min_vol_point, max_sharpe_point, min_vol_weights, max_sharpe_weights

# Load data for all portfolio stocks
with st.spinner(f"Loading data for {len(portfolio_stocks)} stocks..."):
    portfolio_data, portfolio_returns, portfolio_closes = load_portfolio_data(portfolio_stocks, start_date, end_date)

# Portfolio Dashboard page
if page == "Portfolio Dashboard" and portfolio_returns is not None:
    st.header("Portfolio Dashboard")
    
    try:
        # Calculate portfolio return based on weights
        portfolio_return = calculate_portfolio_return(portfolio_returns, weights)
        
        # Display portfolio metrics
        metrics = calculate_portfolio_metrics(portfolio_return)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Annual Return", f"{metrics['Annual Return']:.2%}")
        
        with col2:
            st.metric("Annual Volatility", f"{metrics['Annual Volatility']:.2%}")
        
        with col3:
            st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        
        with col4:
            st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")
        
        with col2:
            st.metric("Calmar Ratio", f"{metrics['Calmar Ratio']:.2f}")
        
        with col3:
            st.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
        
        # Cumulative returns chart
        st.subheader("Cumulative Returns")
        
        cumulative_returns = pd.DataFrame()
        cumulative_returns['Portfolio'] = (1 + portfolio_return).cumprod()
        
        # Add benchmark (S&P 500)
        try:
            benchmark_data = load_stock_data('^GSPC', start_date, end_date)
            if benchmark_data is not None:
                benchmark_returns = benchmark_data['Close'].pct_change().dropna()
                cumulative_returns['S&P 500'] = (1 + benchmark_returns).cumprod()
        except Exception as e:
            st.warning(f"Unable to load benchmark data: {e}")
        
        # Add individual stocks
        for ticker in portfolio_stocks:
            if ticker in portfolio_returns.columns:
                cumulative_returns[ticker] = (1 + portfolio_returns[ticker]).cumprod()
        
        # Plot using Plotly
        fig = go.Figure()
        
        # Add portfolio line (thicker)
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns['Portfolio'],
            mode='lines',
            name='Portfolio',
            line=dict(width=3, color='blue')
        ))
        
        # Add benchmark line (if available)
        if 'S&P 500' in cumulative_returns.columns:
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns['S&P 500'],
                mode='lines',
                name='S&P 500',
                line=dict(width=2.5, color='black')
            ))
        
        # Add individual stock lines
        for ticker in portfolio_stocks:
            if ticker in cumulative_returns.columns:
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns[ticker],
                    mode='lines',
                    name=ticker,
                    line=dict(width=1.5, dash='dash')
                ))
        
        fig.update_layout(
            title='Cumulative Returns',
            xaxis_title='Date',
            yaxis_title='Growth of $1 Invested',
            height=500,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Rolling metrics
        st.subheader("Rolling Performance")
        
        # Options for rolling window
        window_options = {
            '30 Days': 30,
            '90 Days': 90,
            '180 Days': 180,
            '1 Year': 252
        }
        
        selected_window = st.selectbox("Select rolling window period:", list(window_options.keys()))
        window = window_options[selected_window]
        
        # Calculate rolling metrics
        rolling_return = portfolio_return.rolling(window=window).mean() * 252
        rolling_vol = portfolio_return.rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add rolling return
        fig.add_trace(
            go.Scatter(x=rolling_return.index, y=rolling_return, name=f"Rolling {selected_window} Return (Annualized)"),
            secondary_y=False
        )
        
        # Add rolling volatility
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol, name=f"Rolling {selected_window} Volatility (Annualized)"),
            secondary_y=False
        )
        
        # Add rolling Sharpe ratio
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name=f"Rolling {selected_window} Sharpe Ratio"),
            secondary_y=True
        )
        
        # Set titles
        fig.update_layout(
            title=f"Rolling {selected_window} Metrics",
            height=500,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Return / Volatility", secondary_y=False)
        fig.update_yaxes(title_text="Sharpe Ratio", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual stock metrics
        st.subheader("Individual Stock Metrics")
        
        metrics_df = pd.DataFrame(index=portfolio_stocks)
        
        for ticker in portfolio_stocks:
            if ticker in portfolio_returns.columns:
                stock_metrics = calculate_portfolio_metrics(portfolio_returns[ticker])
                metrics_df.loc[ticker, 'Annual Return'] = stock_metrics['Annual Return']
                metrics_df.loc[ticker, 'Annual Volatility'] = stock_metrics['Annual Volatility']
                metrics_df.loc[ticker, 'Sharpe Ratio'] = stock_metrics['Sharpe Ratio']
                metrics_df.loc[ticker, 'Max Drawdown'] = stock_metrics['Max Drawdown']
                metrics_df.loc[ticker, 'Weight'] = weights.get(ticker, 0)
        
        # Format for display
        formatted_df = metrics_df.copy()
        formatted_df['Annual Return'] = formatted_df['Annual Return'].apply(lambda x: f"{x:.2%}")
        formatted_df['Annual Volatility'] = formatted_df['Annual Volatility'].apply(lambda x: f"{x:.2%}")
        formatted_df['Sharpe Ratio'] = formatted_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        formatted_df['Max Drawdown'] = formatted_df['Max Drawdown'].apply(lambda x: f"{x:.2%}")
        formatted_df['Weight'] = formatted_df['Weight'].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(formatted_df)
        
        # Correlation matrix heatmap
        st.subheader("Stock Correlation Matrix")
        
        corr_matrix = portfolio_returns.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title="Correlation of Stock Returns"
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error in Portfolio Dashboard: {e}")
    
elif page == "Portfolio Dashboard" and portfolio_returns is None:
    st.info("Please enter valid stock ticker symbols in your portfolio to view the dashboard.")
    st.info("Popular stock tickers include: AAPL (Apple), MSFT (Microsoft), GOOGL (Google), AMZN (Amazon), META (Meta/Facebook)")

# Stock Analysis page
elif page == "Stock Analysis" and portfolio_data is not None:
    st.header("Individual Stock Analysis")
    
    # Select a stock from the portfolio
    selected_stock = st.selectbox("Select a stock to analyze", portfolio_stocks)
    
    if selected_stock in portfolio_data:
        try:
            stock_data = portfolio_data[selected_stock]
            returns_series = portfolio_returns[selected_stock]
            
            # Display stock information
            col1, col2, col3 = st.columns(3)
            with col1:
                # Fix the format string issue by explicitly converting to float
                try:
                    current_price = float(stock_data['Close'].iloc[-1])
                    prev_price = float(stock_data['Close'].iloc[-2])
                    price_change = (current_price - prev_price) / prev_price
                    st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2%}")
                except Exception as e:
                    st.metric("Current Price", "Error loading price")
                    st.error(f"Error displaying price: {str(e)}")
            
            with col2:
                try:
                    annual_return = ((1 + returns_series.mean()) ** 252 - 1)
                    st.metric("Annual Return", f"{annual_return:.2%}")
                except Exception as e:
                    st.metric("Annual Return", "Error calculating return")
                    st.error(f"Error calculating return: {str(e)}")
            
            with col3:
                try:
                    annual_vol = returns_series.std() * np.sqrt(252)
                    st.metric("Annual Volatility", f"{annual_vol:.2%}")
                except Exception as e:
                    st.metric("Annual Volatility", "Error")
            
            # Price and return charts tabs
            tab1, tab2, tab3 = st.tabs(["Price Chart", "Returns Analysis", "Technical Indicators"])
            
            with tab1:
                st.subheader(f"{selected_stock} Price Chart")
                
                try:
                    # Create Plotly figure for price chart
                    fig = go.Figure()
                    
                    # Add candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=stock_data.index,
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'],
                        name='Candlestick'
                    ))
                    
                    # Add volume as bar chart at the bottom
                    fig.add_trace(go.Bar(
                        x=stock_data.index,
                        y=stock_data['Volume'],
                        name='Volume',
                        marker=dict(color='rgba(0, 0, 255, 0.3)'),
                        opacity=0.3,
                        yaxis='y2'
                    ))
                    
                    # Add moving averages
                    ma_periods = [20, 50, 200]
                    ma_colors = ['blue', 'green', 'red']
                    
                    for period, color in zip(ma_periods, ma_colors):
                        ma = stock_data['Close'].rolling(window=period).mean()
                        fig.add_trace(go.Scatter(
                            x=stock_data.index,
                            y=ma,
                            mode='lines',
                            name=f'{period}-day MA',
                            line=dict(width=1.5, color=color)
                        ))
                    
                    # Update layout with dual y-axis
                    fig.update_layout(
                        title=f'{selected_stock} Stock Price',
                        yaxis_title='Price (USD)',
                        xaxis_title='Date',
                        height=600,
                        template='plotly_white',
                        xaxis_rangeslider_visible=False,
                        yaxis2=dict(
                            title='Volume',
                            overlaying='y',
                            side='right',
                            showgrid=False
                        )
                    )
                    
                    # Allow zooming in specific areas
                    fig.update_layout(
                        xaxis=dict(
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=1, label="1m", step="month", stepmode="backward"),
                                    dict(count=6, label="6m", step="month", stepmode="backward"),
                                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                                    dict(count=1, label="1y", step="year", stepmode="backward"),
                                    dict(step="all")
                                ])
                            ),
                            type="date"
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating price chart: {str(e)}")
                    
            with tab2:
                st.subheader(f"{selected_stock} Returns Analysis")
                
                try:
                    # Create tabs for different returns analyses
                    return_tabs = st.tabs(["Daily Returns", "Return Distribution", "Cumulative Returns"])
                    
                    with return_tabs[0]:
                        # Daily returns line chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=returns_series.index,
                            y=returns_series,
                            mode='lines',
                            name='Daily Returns',
                            line=dict(color='blue')
                        ))
                        
                        fig.update_layout(
                            title=f'{selected_stock} Daily Returns',
                            xaxis_title='Date',
                            yaxis_title='Return (%)',
                            height=400,
                            template='plotly_white'
                        )
                        
                        # Add horizontal line at y=0
                        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with return_tabs[1]:
                        # Daily returns histogram
                        fig = go.Figure()
                        
                        # Convert returns to numeric values, just to be safe
                        returns_numeric = pd.to_numeric(returns_series, errors='coerce').dropna()
                        
                        fig.add_trace(go.Histogram(
                            x=returns_numeric,
                            nbinsx=50,
                            name='Daily Returns',
                            marker_color='blue',
                            opacity=0.7
                        ))
                        
                        # Add a normal distribution curve for comparison
                        mean = returns_numeric.mean()
                        std = returns_numeric.std()
                        x_range = np.linspace(min(returns_numeric), max(returns_numeric), 100)
                        y_norm = np.exp(-(x_range - mean)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))
                        y_norm = y_norm * len(returns_numeric) * (max(returns_numeric) - min(returns_numeric)) / 50
                        
                        fig.add_trace(go.Scatter(
                            x=x_range,
                            y=y_norm,
                            mode='lines',
                            name='Normal Distribution',
                            line=dict(color='red', width=2)
                        ))
                        
                        fig.update_layout(
                            title=f'{selected_stock} Daily Returns Distribution',
                            xaxis_title='Daily Return',
                            yaxis_title='Frequency',
                            height=400,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics about the distribution
                        st.subheader("Return Distribution Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean", f"{returns_numeric.mean():.4f}")
                        with col2:
                            st.metric("Std Dev", f"{returns_numeric.std():.4f}")
                        with col3:
                            st.metric("Skewness", f"{returns_numeric.skew():.4f}")
                        with col4:
                            st.metric("Kurtosis", f"{returns_numeric.kurtosis():.4f}")
                    
                    with return_tabs[2]:
                        # Cumulative returns
                        cumulative_returns = (1 + returns_numeric).cumprod()
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=cumulative_returns.index,
                            y=cumulative_returns,
                            mode='lines',
                            name='Cumulative Return',
                            line=dict(width=2, color='green')
                        ))
                        
                        # Add a reference line for buy-and-hold strategy
                        fig.add_trace(go.Scatter(
                            x=[cumulative_returns.index[0], cumulative_returns.index[-1]],
                            y=[1, cumulative_returns.iloc[-1]],
                            mode='lines',
                            name='Linear Growth',
                            line=dict(width=1, color='red', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f'{selected_stock} Cumulative Returns',
                            xaxis_title='Date',
                            yaxis_title='Growth of $1 Invested',
                            height=400,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate and display annualized return
                        days = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days
                        years = days / 365.25
                        annualized_return = (cumulative_returns.iloc[-1] ** (1 / years)) - 1
                        
                        st.metric("Annualized Return (CAGR)", f"{annualized_return:.2%}")
                
                except Exception as e:
                    st.error(f"Error creating returns analysis: {str(e)}")
            
            with tab3:
                st.subheader(f"{selected_stock} Technical Indicators")
                
                try:
                    # Calculate technical indicators
                    df = stock_data.copy()
                    
                    # RSI (Relative Strength Index)
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    RS = gain / loss
                    df['RSI'] = 100 - (100 / (1 + RS))
                    
                    # MACD (Moving Average Convergence Divergence)
                    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
                    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
                    df['MACD'] = ema12 - ema26
                    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
                    
                    # Bollinger Bands
                    df['SMA20'] = df['Close'].rolling(window=20).mean()
                    df['STD20'] = df['Close'].rolling(window=20).std()
                    df['UpperBand'] = df['SMA20'] + (df['STD20'] * 2)
                    df['LowerBand'] = df['SMA20'] - (df['STD20'] * 2)
                    
                    # ATR (Average True Range)
                    df['TR'] = np.maximum(
                        np.maximum(
                            df['High'] - df['Low'],
                            abs(df['High'] - df['Close'].shift(1))
                        ),
                        abs(df['Low'] - df['Close'].shift(1))
                    )
                    df['ATR'] = df['TR'].rolling(window=14).mean()
                    
                    # Create tabs for different indicators
                    indicator_tabs = st.tabs(["RSI", "MACD", "Bollinger Bands"])
                    
                    with indicator_tabs[0]:
                        # RSI Chart
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                           vertical_spacing=0.05, row_heights=[0.7, 0.3])
                        
                        # Price chart
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='blue')
                        ), row=1, col=1)
                        
                        # RSI
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['RSI'],
                            mode='lines',
                            name='RSI',
                            line=dict(color='purple')
                        ), row=2, col=1)
                        
                        # Add horizontal lines at 30 and 70
                        fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=2, col=1)
                        fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
                        
                        fig.update_layout(
                            title='Relative Strength Index (RSI)',
                            height=600,
                            template='plotly_white'
                        )
                        
                        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
                        fig.update_yaxes(title_text="RSI", row=2, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with indicator_tabs[1]:
                        # MACD Chart
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                           vertical_spacing=0.05, row_heights=[0.7, 0.3])
                        
                        # Price chart
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='blue')
                        ), row=1, col=1)
                        
                        # MACD
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['MACD'],
                            mode='lines',
                            name='MACD',
                            line=dict(color='blue')
                        ), row=2, col=1)
                        
                        # MACD Signal
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['MACD_signal'],
                            mode='lines',
                            name='Signal Line',
                            line=dict(color='red')
                        ), row=2, col=1)
                        
                        # MACD Histogram
                        colors = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
                        fig.add_trace(go.Bar(
                            x=df.index,
                            y=df['MACD_hist'],
                            name='Histogram',
                            marker_color=colors
                        ), row=2, col=1)
                        
                        fig.update_layout(
                            title='Moving Average Convergence Divergence (MACD)',
                            height=600,
                            template='plotly_white'
                        )
                        
                        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
                        fig.update_yaxes(title_text="MACD", row=2, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with indicator_tabs[2]:
                        # Bollinger Bands Chart
                        fig = go.Figure()
                        
                        # Price chart
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='blue')
                        ))
                        
                        # Upper Band
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['UpperBand'],
                            mode='lines',
                            name='Upper Band (2σ)',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Middle Band (SMA)
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['SMA20'],
                            mode='lines',
                            name='SMA (20)',
                            line=dict(color='green')
                        ))
                        
                        # Lower Band
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['LowerBand'],
                            mode='lines',
                            name='Lower Band (2σ)',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Fill between upper and lower bands
                        fig.add_trace(go.Scatter(
                            x=df.index.tolist() + df.index.tolist()[::-1],
                            y=df['UpperBand'].tolist() + df['LowerBand'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(0,100,80,0.2)',
                            line=dict(width=0),
                            name='Band Range'
                        ))
                        
                        fig.update_layout(
                            title='Bollinger Bands',
                            height=600,
                            template='plotly_white'
                        )
                        
                        fig.update_yaxes(title_text="Price (USD)")
                        
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating technical indicators: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error analyzing {selected_stock}: {str(e)}")
            st.info("This could be due to data quality issues or missing values. Try selecting a different stock or date range.")
    else:
        st.warning(f"Data for {selected_stock} is not available. Please select another stock.")
elif page == "Stock Analysis" and portfolio_data is None:
    st.info("Please enter valid stock ticker symbols in your portfolio to analyze individual stocks.")

# ML Predictions page
elif page == "ML Predictions" and portfolio_data is not None:
    st.header("Machine Learning Stock Predictions")
    
    # Select a stock for predictions
    selected_stock = st.selectbox("Select a stock for prediction analysis:", portfolio_stocks)
    
    if selected_stock in portfolio_data:
        try:
            # Create features for ML models
            window_sizes = [5, 10, 20, 50]
            target_days_options = [1, 3, 5, 10, 21]
            
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox(
                    "Select prediction model:",
                    ["Random Forest", "XGBoost", "ARIMA"]
                )
            
            with col2:
                if model_type in ["Random Forest", "XGBoost"]:
                    target_days = st.selectbox(
                        "Predict returns over how many days:",
                        target_days_options
                    )
                    
                    test_size = st.slider(
                        "Test set size (% of data):",
                        min_value=0.1,
                        max_value=0.5,
                        value=0.2,
                        step=0.05
                    )
                else:  # ARIMA
                    p = st.number_input("ARIMA p (AR order):", 0, 10, 5)
                    d = st.number_input("ARIMA d (differencing):", 0, 2, 1)
                    q = st.number_input("ARIMA q (MA order):", 0, 10, 0)
                    forecast_days = st.number_input("Forecast days:", 5, 252, 30)
            
            if st.button(f"Train {model_type} Model"):
                try:
                    with st.spinner(f"Training {model_type} model for {selected_stock}..."):
                        # Based on model type, train the appropriate model
                        if model_type == "Random Forest" or model_type == "XGBoost":
                            # Create features with more robust error handling
                            try:
                                features_df = create_features(
                                    portfolio_data, 
                                    selected_stock, 
                                    window_sizes=window_sizes, 
                                    target_days=target_days
                                )
                                
                                # Display feature dataframe preview
                                st.subheader("Feature Engineering Preview")
                                st.write(features_df.tail().style.format(precision=4))
                                
                                # Check for valid data
                                if features_df.empty:
                                    st.error("Not enough data to create features. Try a different stock or extend the date range.")
                                    
                                
                                # Train model
                                model_name = "random_forest" if model_type == "Random Forest" else "xgboost"
                                model, predictions_df, metrics, features_df, scaler = train_ml_model(
                                    features_df,
                                    model_type=model_name,
                                    target_days=target_days,
                                    test_size=test_size
                                )
                                
                                # Define feature_cols here before using it later
                                exclude_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', f'target_{target_days}d']
                                feature_cols = [col for col in features_df.columns if col not in exclude_cols]
                                
                                # Display model performance
                                st.subheader("Model Performance Metrics")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Mean Squared Error", f"{metrics['MSE']:.6f}")
                                with col2:
                                    st.metric("Mean Absolute Error", f"{metrics['MAE']:.6f}")
                                with col3:
                                    st.metric("R² Score", f"{metrics['R2']:.4f}")
                                
                                # Plot actual vs predicted values
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=predictions_df['Date'],
                                    y=predictions_df['Actual'],
                                    mode='lines',
                                    name='Actual Returns',
                                    line=dict(color='blue')
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=predictions_df['Date'],
                                    y=predictions_df['Predicted'],
                                    mode='lines',
                                    name='Predicted Returns',
                                    line=dict(color='red')
                                ))
                                
                                fig.update_layout(
                                    title=f"{model_type} Model: Actual vs Predicted {target_days}-Day Returns",
                                    xaxis_title="Date",
                                    yaxis_title=f"{target_days}-Day Return",
                                    height=500,
                                    template='plotly_white'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Feature importance
                                if features_df is not None:
                                    st.subheader("Feature Importance")
                                    
                                    # Bar chart of feature importance
                                    fig = px.bar(
                                        features_df.head(15),
                                        x='Importance',
                                        y='Feature',
                                        orientation='h',
                                        title="Top 15 Most Important Features"
                                    )
                                    
                                    fig.update_layout(height=500)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Future prediction
                                st.subheader(f"Predict Future {target_days}-Day Return")
                                
                                # Get latest data for prediction
                                latest_data = features_df.iloc[-1:][feature_cols]
                                
                                # Make prediction
                                predicted_return = predict_future_returns(model, scaler, latest_data, target_days)
                                
                                # Current price and predicted price
                                current_price = portfolio_data[selected_stock]['Close'].iloc[-1]
                                predicted_price = current_price * (1 + predicted_return)
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "Current Price", 
                                        f"${current_price:.2f}"
                                    )
                                
                                with col2:
                                    st.metric(
                                        f"Predicted {target_days}-Day Return", 
                                        f"{predicted_return:.2%}"
                                    )
                                
                                with col3:
                                    st.metric(
                                        f"Predicted Price (in {target_days} days)", 
                                        f"${predicted_price:.2f}",
                                        f"{predicted_return:.2%}"
                                    )
                                
                                # Add disclaimer
                                st.warning("Disclaimer: These predictions are for educational purposes only. Do not use them for actual investment decisions.")
                                
                            except Exception as e:
                                st.error(f"Error in feature engineering or model training: {str(e)}")
                                st.info("Try using a different stock with more complete data, or extending your date range.")
                                
                                
                        elif model_type == "ARIMA":
                            # Train ARIMA model
                            model, forecast_df, metrics = train_arima_model(
                                portfolio_data, 
                                selected_stock, 
                                p=p, d=d, q=q,
                                forecast_days=forecast_days
                            )
                            
                            if model is not None:
                                # Display model performance
                                st.subheader("ARIMA Model Performance")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("AIC", f"{metrics['AIC']:.2f}")
                                with col2:
                                    st.metric("BIC", f"{metrics['BIC']:.2f}")
                                
                                # Plot historical data and forecast
                                fig = go.Figure()
                                
                                # Historical data
                                fig.add_trace(go.Scatter(
                                    x=portfolio_data[selected_stock].index,
                                    y=portfolio_data[selected_stock]['Close'],
                                    mode='lines',
                                    name='Historical Price',
                                    line=dict(color='blue')
                                ))
                                
                                # Forecast
                                fig.add_trace(go.Scatter(
                                    x=forecast_df['Date'],
                                    y=forecast_df['Forecast'],
                                    mode='lines',
                                    name='Forecast',
                                    line=dict(color='red')
                                ))
                                
                                # Add confidence interval
                                # Note: ARIMA model results have confidence intervals but would need additional code
                                
                                fig.update_layout(
                                    title=f"ARIMA({p},{d},{q}) Forecast for {selected_stock}",
                                    xaxis_title="Date",
                                    yaxis_title="Price (USD)",
                                    height=500,
                                    template='plotly_white'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display forecast table
                                st.subheader("Price Forecast")
                                
                                # Format the forecast dataframe
                                forecast_display = forecast_df.copy()
                                forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
                                forecast_display['Forecast'] = forecast_display['Forecast'].round(2)
                                
                                st.dataframe(forecast_display)
                                
                                # Add disclaimer
                                st.warning("Disclaimer: These forecasts are for educational purposes only. Do not use them for actual investment decisions.")
                            else:
                                st.error("Error: ARIMA model could not be trained. Try different parameters or another stock.")
                
                except Exception as e:
                    st.error(f"Error in ML Predictions: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error in ML Predictions: {str(e)}")
    else:
        st.warning(f"Data for {selected_stock} is not available. Please select another stock.")
elif page == "ML Predictions" and portfolio_data is None:
    st.info("Please enter valid stock ticker symbols in your portfolio to use ML predictions.")

# Portfolio Optimization page
elif page == "Portfolio Optimization" and portfolio_data is not None:
    st.header("Portfolio Optimization")
    
    try:
        # Check if we have enough data
        if len(portfolio_returns.columns) < 2:
            st.warning("Portfolio optimization requires at least 2 stocks. Please add more stocks to your portfolio.")
        else:
            # Get mean returns and covariance matrix
            mean_returns = portfolio_returns.mean()
            cov_matrix = portfolio_returns.cov()
            
            # Display current portfolio info
            st.subheader("Current Portfolio")
            
            # Calculate current portfolio performance
            portfolio_return = calculate_portfolio_return(portfolio_returns, weights)
            metrics = calculate_portfolio_metrics(portfolio_return)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Annual Return", f"{metrics['Annual Return']:.2%}")
            
            with col2:
                st.metric("Annual Volatility", f"{metrics['Annual Volatility']:.2%}")
            
            with col3:
                st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
            
            # Display current weights
            current_weights_df = pd.DataFrame({
                'Stock': list(weights.keys()),
                'Weight': list(weights.values())
            })
            
            # Format for display
            current_weights_df['Weight'] = current_weights_df['Weight'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(current_weights_df)
            
            # Portfolio optimization section
            st.subheader("Portfolio Optimization")
            
            optimization_type = st.radio(
                "Optimization Strategy:",
                ["Maximum Sharpe Ratio", "Minimum Volatility", "Efficient Frontier"]
            )
            
            if optimization_type == "Maximum Sharpe Ratio":
                # Calculate optimal weights for maximum Sharpe ratio
                max_sharpe_weights, max_sharpe_return, max_sharpe_std, max_sharpe_sharpe = max_sharpe_ratio(
                    mean_returns, cov_matrix, portfolio_returns.columns
                )
                
                # Display optimal portfolio
                st.subheader("Optimal Portfolio (Maximum Sharpe Ratio)")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Annual Return", f"{max_sharpe_return:.2%}")
                
                with col2:
                    st.metric("Annual Volatility", f"{max_sharpe_std:.2%}")
                
                with col3:
                    st.metric("Sharpe Ratio", f"{max_sharpe_sharpe:.2f}")
                
                # Display optimal weights
                optimal_weights_df = pd.DataFrame({
                    'Stock': list(max_sharpe_weights.keys()),
                    'Current Weight': [weights.get(ticker, 0) for ticker in max_sharpe_weights.keys()],
                    'Optimal Weight': list(max_sharpe_weights.values())
                })
                
                # Format for display
                optimal_weights_df['Current Weight'] = optimal_weights_df['Current Weight'].apply(lambda x: f"{x:.2%}")
                optimal_weights_df['Optimal Weight'] = optimal_weights_df['Optimal Weight'].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(optimal_weights_df)
                
                # Plot weights comparison
                fig = go.Figure()
                
                # Current weights
                fig.add_trace(go.Bar(
                    x=list(weights.keys()),
                    y=list(weights.values()),
                    name='Current',
                    marker_color='blue'
                ))
                
                # Optimal weights
                fig.add_trace(go.Bar(
                    x=list(max_sharpe_weights.keys()),
                    y=list(max_sharpe_weights.values()),
                    name='Optimal',
                    marker_color='green'
                ))
                
                fig.update_layout(
                    title='Current vs. Optimal Portfolio Weights',
                    xaxis_title='Stock',
                    yaxis_title='Weight',
                    barmode='group',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display improvement
                sharpe_improvement = max_sharpe_sharpe - metrics['Sharpe Ratio']
                return_improvement = max_sharpe_return - metrics['Annual Return']
                
                st.subheader("Potential Improvement")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Return Improvement", f"{return_improvement:.2%}")
                
                with col2:
                    st.metric("Sharpe Ratio Improvement", f"{sharpe_improvement:.2f}")
                
            elif optimization_type == "Minimum Volatility":
                # Calculate optimal weights for minimum volatility
                min_vol_weights, min_vol_return, min_vol_std, min_vol_sharpe = min_volatility(
                    mean_returns, cov_matrix, portfolio_returns.columns
                )
                
                # Display optimal portfolio
                st.subheader("Optimal Portfolio (Minimum Volatility)")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Annual Return", f"{min_vol_return:.2%}")
                
                with col2:
                    st.metric("Annual Volatility", f"{min_vol_std:.2%}")
                
                with col3:
                    st.metric("Sharpe Ratio", f"{min_vol_sharpe:.2f}")
                
                # Display optimal weights
                optimal_weights_df = pd.DataFrame({
                    'Stock': list(min_vol_weights.keys()),
                    'Current Weight': [weights.get(ticker, 0) for ticker in min_vol_weights.keys()],
                    'Optimal Weight': list(min_vol_weights.values())
                })
                
                # Format for display
                optimal_weights_df['Current Weight'] = optimal_weights_df['Current Weight'].apply(lambda x: f"{x:.2%}")
                optimal_weights_df['Optimal Weight'] = optimal_weights_df['Optimal Weight'].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(optimal_weights_df)
                
                # Plot weights comparison
                fig = go.Figure()
                
                # Current weights
                fig.add_trace(go.Bar(
                    x=list(weights.keys()),
                    y=list(weights.values()),
                    name='Current',
                    marker_color='blue'
                ))
                
                # Optimal weights
                fig.add_trace(go.Bar(
                    x=list(min_vol_weights.keys()),
                    y=list(min_vol_weights.values()),
                    name='Optimal',
                    marker_color='green'
                ))
                
                fig.update_layout(
                    title='Current vs. Optimal Portfolio Weights',
                    xaxis_title='Stock',
                    yaxis_title='Weight',
                    barmode='group',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display improvement
                vol_improvement = metrics['Annual Volatility'] - min_vol_std
                
                st.subheader("Potential Improvement")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Volatility Reduction", f"{vol_improvement:.2%}")
                
                with col2:
                    sharpe_improvement = min_vol_sharpe - metrics['Sharpe Ratio']
                    st.metric("Sharpe Ratio Improvement", f"{sharpe_improvement:.2f}")
                
            elif optimization_type == "Efficient Frontier":
                # Calculate efficient frontier
                efficient_df, min_vol_point, max_sharpe_point, min_vol_weights, max_sharpe_weights = efficient_frontier(
                    mean_returns, cov_matrix, portfolio_returns.columns
                )
                
                # Plot efficient frontier
                fig = go.Figure()
                
                # Efficient frontier
                fig.add_trace(go.Scatter(
                    x=efficient_df['Volatility'],
                    y=efficient_df['Return'],
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='blue', width=2)
                ))
                
                # Minimum volatility portfolio
                fig.add_trace(go.Scatter(
                    x=[min_vol_point['Volatility']],
                    y=[min_vol_point['Return']],
                    mode='markers',
                    name='Minimum Volatility',
                    marker=dict(color='green', size=12, symbol='star')
                ))
                
                # Maximum Sharpe ratio portfolio
                fig.add_trace(go.Scatter(
                    x=[max_sharpe_point['Volatility']],
                    y=[max_sharpe_point['Return']],
                    mode='markers',
                    name='Maximum Sharpe Ratio',
                    marker=dict(color='red', size=12, symbol='star')
                ))
                
                # Current portfolio
                fig.add_trace(go.Scatter(
                    x=[metrics['Annual Volatility']],
                    y=[metrics['Annual Return']],
                    mode='markers',
                    name='Current Portfolio',
                    marker=dict(color='black', size=10)
                ))
                
                # Individual stocks
                for ticker in portfolio_returns.columns:
                    stock_return = portfolio_returns[ticker].mean() * 252
                    stock_vol = portfolio_returns[ticker].std() * np.sqrt(252)
                    
                    fig.add_trace(go.Scatter(
                        x=[stock_vol],
                        y=[stock_return],
                        mode='markers',
                        name=ticker,
                        marker=dict(size=8)
                    ))
                
                fig.update_layout(
                    title='Efficient Frontier',
                    xaxis_title='Annual Volatility',
                    yaxis_title='Annual Return',
                    height=600,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display optimal portfolios
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Minimum Volatility Portfolio")
                    
                    st.metric("Annual Return", f"{min_vol_point['Return']:.2%}")
                    st.metric("Annual Volatility", f"{min_vol_point['Volatility']:.2%}")
                    st.metric("Sharpe Ratio", f"{min_vol_point['Sharpe']:.2f}")
                    
                    # Display weights
                    min_vol_weights_df = pd.DataFrame({
                        'Stock': list(min_vol_weights.keys()),
                        'Weight': list(min_vol_weights.values())
                    })
                    
                    min_vol_weights_df['Weight'] = min_vol_weights_df['Weight'].apply(lambda x: f"{x:.2%}")
                    
                    st.dataframe(min_vol_weights_df)
                
                with col2:
                    st.subheader("Maximum Sharpe Ratio Portfolio")
                    
                    st.metric("Annual Return", f"{max_sharpe_point['Return']:.2%}")
                    st.metric("Annual Volatility", f"{max_sharpe_point['Volatility']:.2%}")
                    st.metric("Sharpe Ratio", f"{max_sharpe_point['Sharpe']:.2f}")
                    
                    # Display weights
                    max_sharpe_weights_df = pd.DataFrame({
                        'Stock': list(max_sharpe_weights.keys()),
                        'Weight': list(max_sharpe_weights.values())
                    })
                    
                    max_sharpe_weights_df['Weight'] = max_sharpe_weights_df['Weight'].apply(lambda x: f"{x:.2%}")
                    
                    st.dataframe(max_sharpe_weights_df)
                
                # Allow user to choose a point on the efficient frontier
                st.subheader("Custom Portfolio on Efficient Frontier")
                
                target_return = st.slider(
                    "Target Annual Return",
                    min_value=float(min_vol_point['Return'] - 0.02),
                    max_value=float(max_sharpe_point['Return'] + 0.02),
                    value=float((min_vol_point['Return'] + max_sharpe_point['Return']) / 2),
                    step=0.01,
                    format="%.2f"
                )
                
                # Find the closest portfolio on the efficient frontier
                efficient_df['Return_Diff'] = efficient_df['Return'].apply(lambda x: abs(x - target_return))
                closest_idx = efficient_df['Return_Diff'].idxmin()
                closest_portfolio = efficient_df.loc[closest_idx]
                
                st.write(f"Closest portfolio on the efficient frontier with return {closest_portfolio['Return']:.2%} and volatility {closest_portfolio['Volatility']:.2%}")
                
                # Note: To get the exact weights for this custom point, we would need additional optimization
                # This is simplified for the dashboard example
                
                # In a full implementation, we would solve for weights that give the target return with minimum variance
    
    except Exception as e:
        st.error(f"Error in Portfolio Optimization: {str(e)}")
elif page == "Portfolio Optimization" and portfolio_data is None:
    st.info("Please enter valid stock ticker symbols in your portfolio to use portfolio optimization.")

# Disclaimer
st.sidebar.markdown("---")
st.sidebar.info("""
**Disclaimer:** This application is for educational purposes only. The information provided should not be considered financial advice.
Always consult with a qualified financial advisor before making investment decisions.
""")

if __name__ == "__main__":
    # This will run the Streamlit app when executed directly
    pass