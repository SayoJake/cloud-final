import pandas as pd
import os

stock_dir = "Data/archive/Data/Stocks"
files = os.listdir(stock_dir)

# Filter for only ".txt" files
txt_files = [f for f in files if f.endswith('.txt')]

# Just pick the first 20 files (or fewer if you want)
txt_files = txt_files[21:40]

dfs = []
for f in txt_files:
    path = os.path.join(stock_dir, f)
    df = pd.read_csv(path)
    # File format: <ticker>.us.txt, extract ticker name before ".us.txt"
    ticker = f.split('.us.txt')[0].upper()
    df['Ticker'] = ticker
    # Expected columns: Date,Open,High,Low,Close,Volume,OpenInt
    # Make sure to have consistent schema: Date (YYYY-MM-DD)
    # Convert Date column to string if needed
    df['Date'] = pd.to_datetime(df['Date'])
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# Save combined dataset
combined_df.to_csv('combined_21to40_stocks.csv', index=False)
