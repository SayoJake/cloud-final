import os
from flask import Flask, request, render_template, redirect, url_for, flash, session
from google.cloud import bigquery
import pandas as pd
import plotly.express as px
from io import StringIO
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # For flash messages

client = bigquery.Client(project="cloud-final-finance")

DATASET_ID = "financial_data"
TABLE_ID = "stock_data"

@app.before_request
def require_login():
    allowed_routes = ['signup', 'login', 'index']
    if 'username' not in session and request.endpoint not in allowed_routes:
        return redirect(url_for('login'))

@app.route('/')
def index():
    """Root route. If logged in, go to home; otherwise, go to login."""
    if 'username' in session:
        return redirect(url_for('home'))
    else:
        return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        session['username'] = username
        flash(f"User {username} signed up with email {email}")
        return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password and username == password:
            session['username'] = username
            flash("Logged in successfully!")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials, please try again.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out successfully!")
    return redirect(url_for('login'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/search', methods=['GET'])
def search():
    return render_template('search.html')

@app.route('/datapull', methods=['GET'])
def datapull():
    ticker = request.args.get('ticker', '').upper()
    if not ticker:
        ticker = "AAPL"
    query = f"""
    SELECT * FROM `{DATASET_ID}.{TABLE_ID}`
    WHERE Ticker = '{ticker}'
    ORDER BY Date
    LIMIT 100
    """
    try:
        df = client.query(query).to_dataframe()
        if df.empty:
            return render_template('error.html', message=f"No data found for ticker {ticker}")
        return df.to_html(index=False)
    except Exception as e:
        return render_template('error.html', message=str(e))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            flash("No file selected")
            return redirect(request.url)
        try:
            dataset_ref = client.dataset(DATASET_ID)
            client.get_dataset(dataset_ref)

            contents = file.read().decode('utf-8')
            new_df = pd.read_csv(StringIO(contents))
            table_ref = dataset_ref.table(TABLE_ID)
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
            job = client.load_table_from_dataframe(new_df, table_ref, job_config=job_config)
            job.result()
            flash("Data uploaded successfully!")
            return redirect(url_for('upload'))
        except Exception as e:
            return render_template('error.html', message=f"Error processing file: {str(e)}")

    return render_template('upload.html')

def get_all_tickers():
    """Helper function to get all tickers from BigQuery."""
    distinct_tickers_query = f"SELECT DISTINCT Ticker FROM `{DATASET_ID}.{TABLE_ID}`"
    tickers_df = client.query(distinct_tickers_query).to_dataframe()
    tickers = tickers_df['Ticker'].tolist()
    return tickers

@app.route('/dashboard', methods=['GET'])
def dashboard():
    tickers = get_all_tickers()
    print("Tickers found in dashboard:", tickers)
    if not tickers:
        return render_template('error.html', message="No tickers available in the dataset.")

    selected_ticker = request.args.get('ticker', '').upper()
    if not selected_ticker or selected_ticker not in tickers:
        selected_ticker = tickers[0]

    query = f"SELECT * FROM `{DATASET_ID}.{TABLE_ID}` WHERE Ticker='{selected_ticker}' ORDER BY Date"
    try:
        df = client.query(query).to_dataframe()
        if df.empty:
            return render_template('error.html', message=f"No data found for ticker {selected_ticker}.")

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        fig = px.line(df, x='Date', y='Close', title=f'{selected_ticker} Closing Price Trends')
        graph_html = fig.to_html(full_html=False)

        scatter_fig = px.scatter(df, x='Volume', y='Close', title=f'{selected_ticker} Volume vs. Close Price Correlation')
        volume_correlation_html = scatter_fig.to_html(full_html=False)

        df['Month'] = df['Date'].dt.to_period('M').astype(str)
        monthly_stats = df.groupby('Month')['Close'].agg(['mean', 'min', 'max']).reset_index()
        bar_fig = px.bar(monthly_stats, x='Month', y=['mean', 'min', 'max'],
                         title=f'{selected_ticker} Monthly Close Price Stats',
                         labels={'value': 'Close Price', 'variable': 'Stat'})
        monthly_performance_html = bar_fig.to_html(full_html=False)

        max_close_date = df.loc[df['Close'].idxmax(), 'Date'].strftime('%Y-%m-%d')
        insights = f"The highest close price for {selected_ticker} was {df['Close'].max():.2f} on {max_close_date}."

        df['Change (%)'] = df['Close'].pct_change() * 100
        top_gainers = df.nlargest(5, 'Change (%)')[['Date', 'Close', 'Change (%)']]
        top_losers = df.nsmallest(5, 'Change (%)')[['Date', 'Close', 'Change (%)']]
        movers_table = pd.concat([top_gainers, top_losers]).to_html(index=False, classes="table table-striped")

        stats_table = df[['Close', 'Volume']].describe().to_html(classes="table table-bordered")

        return render_template(
            'dashboard.html',
            tickers=tickers,
            selected_ticker=selected_ticker,
            graph_html=graph_html,
            volume_correlation_html=volume_correlation_html,
            monthly_performance_html=monthly_performance_html,
            insights=insights,
            top_movers_table=movers_table,
            stats_table=stats_table,
        )
    except Exception as e:
        return render_template('error.html', message=str(e))

@app.route('/mlmodel')
def mlmodel():
    tickers = get_all_tickers()
    if not tickers:
        return render_template('error.html', message="No tickers available for ML.")

    # Allow selecting ticker
    selected_ticker = request.args.get('ticker', '').upper()
    if not selected_ticker or selected_ticker not in tickers:
        selected_ticker = tickers[0]

    query = f"SELECT * FROM `{DATASET_ID}.{TABLE_ID}` WHERE Ticker='{selected_ticker}' ORDER BY Date"
    try:
        df = client.query(query).to_dataframe()
        if df.empty:
            return render_template('error.html', message=f"No data found for ticker {selected_ticker}.")

        df['Date'] = pd.to_datetime(df['Date'])
        df['Previous_Close'] = df['Close'].shift(1)
        df = df.dropna()
        X = df[['Previous_Close']]
        y = df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        return render_template('trend.html',  # Reuse trend template or create mlmodel.html
                               ticker=selected_ticker,
                               predictions=[],
                               score=f"{score:.2f}",
                               ml_page=True,
                               info="Close Price Prediction Model")
    except Exception as e:
        return render_template('error.html', message=str(e))

@app.route('/trend_prediction', methods=['GET'])
def trend_prediction():
    tickers = get_all_tickers()
    if not tickers:
        return render_template('error.html', message="No tickers available for Trend Prediction.")

    selected_ticker = request.args.get('ticker', '').upper()
    if not selected_ticker or selected_ticker not in tickers:
        selected_ticker = tickers[0]

    query = f"SELECT * FROM `{DATASET_ID}.{TABLE_ID}` WHERE Ticker='{selected_ticker}' ORDER BY Date"
    try:
        # Load and preprocess data
        df = client.query(query).to_dataframe()
        if df.empty:
            return render_template('error.html', message=f"No data found for ticker {selected_ticker}.")

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df['Previous_Close'] = df['Close'].shift(1)

        # Add rolling features
        df['5_day_avg'] = df['Close'].rolling(window=5).mean()
        df['10_day_avg'] = df['Close'].rolling(window=10).mean()
        df['5_day_volatility'] = df['Close'].rolling(window=5).std()
        df = df.dropna()

        if df.empty:
            return render_template('error.html', message="Not enough data for predictions.")

        # Prepare data for ML
        X = df[['Previous_Close', '5_day_avg', '10_day_avg', '5_day_volatility']]
        y = df['Close']

        # Train Gradient Boosting Model
        gbr_model = GradientBoostingRegressor()
        gbr_model.fit(X, y)

        # Train Random Forest Model
        rf_model = RandomForestRegressor()
        rf_model.fit(X, y)

        # Predict next 3 days
        from datetime import timedelta
        last_date = df['Date'].iloc[-1]
        last_row = X.iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 4)]
        gbr_predictions, rf_predictions = [], []

        for _ in range(3):
            gbr_pred = gbr_model.predict([last_row])[0]
            rf_pred = rf_model.predict([last_row])[0]
            gbr_predictions.append(gbr_pred)
            rf_predictions.append(rf_pred)

            # Update rolling features
            last_row['Previous_Close'] = gbr_pred
            last_row['5_day_avg'] = (last_row['5_day_avg'] * 4 + gbr_pred) / 5
            last_row['10_day_avg'] = (last_row['10_day_avg'] * 9 + gbr_pred) / 10
            last_row['5_day_volatility'] = np.std([last_row['Previous_Close'], gbr_pred])

        # Visualization
        fig = px.line(df, x='Date', y='Close', title=f"{selected_ticker} Historical Data and Predictions")
        future_df = pd.DataFrame({'Date': future_dates, 'GBR_Prediction': gbr_predictions, 'RF_Prediction': rf_predictions})
        for col in ['GBR_Prediction', 'RF_Prediction']:
            fig.add_scatter(x=future_df['Date'], y=future_df[col], mode='lines', name=col)

        # Descriptive Statistics
        stats = df[['Close']].describe().to_html(classes="table table-striped", float_format="%.2f")

        # Volatility Analysis
        volatility = df['5_day_volatility'].mean()

        return render_template(
            'trend.html',
            ticker=selected_ticker,
            prediction_chart=fig.to_html(full_html=False),
            tickers=tickers,
            selected_ticker=selected_ticker,
            future_table=future_df.to_html(index=False, classes="table table-striped", float_format="%.2f"),
            stats=stats,
            volatility=f"Avg. Volatility (5-day): {volatility:.2f}"
        )
    except Exception as e:
        return render_template('error.html', message=str(e))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
