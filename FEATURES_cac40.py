import pandas as pd
import numpy as np

# Load rebalancing dates and compositions
index_compo = pd.read_excel("INDEX_COMPO.xlsx", sheet_name="CAC40")
index_compo["Rebalancing Date"] = pd.to_datetime(index_compo["Rebalancing Date"], dayfirst=True)

# Load stock data
xl = pd.ExcelFile("french_stocks_close_volume.xlsx")
ticker_data = {}

for ticker in xl.sheet_names:
    df = xl.parse(ticker)
    df.columns = ["Date", "Close", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = df["Close"].astype(str).str.replace(",", ".").astype(float)
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df.set_index("Date", inplace=True)
    
    # Calculate daily returns (needed for volatility)
    df["Return"] = df["Close"].pct_change()
    
    ticker_data[ticker] = df

# Load index and commodity spot data
spot_xl = pd.ExcelFile("DATA_spot_prices_ind_cty.xlsx")

# Define the mapping of tickers to feature names
index_tickers = {
    "^FCHI": "CAC40",
    "FTSEMIB.MI": "FTSEMIB",
    "^GSPC": "SP500",
    "^FTSE": "FTSE100",
    "^SSMI": "SMI",
    "^HSCE": "HSCEI",
    "^N225": "Nikkei225",
    "^RUT": "Russell2000",
    "IWDA.L": "MSCI_World",
    "EEM": "MSCI_Emerging",
    "^DJI": "DowJones",
    "^GDAXI": "DAX"
}

index_data = {}
for ticker, name in index_tickers.items():
    df = spot_xl.parse(name)
    df.columns = ["Date", "Close"]
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df["Close"] = df["Close"].astype(float)
    index_data[name] = df

# Load Fed and ECB rates
fed_rate = spot_xl.parse("FED")
fed_rate.columns = ["Date", "Fed Funds Rate (%)"]
fed_rate["Date"] = pd.to_datetime(fed_rate["Date"])
fed_rate.set_index("Date", inplace=True)

ecb_rate = spot_xl.parse("BCE")
ecb_rate.columns = ["Date", "ECB Interest Rate (%)"]
ecb_rate["Date"] = pd.to_datetime(ecb_rate["Date"])
ecb_rate.set_index("Date", inplace=True)

# Load commodities
commodity_names = ["Gold", "Silver", "NaturalGas"]
commodity_data = {}
for name in commodity_names:
    df = spot_xl.parse(name)
    df.columns = ["Date", "Close"]
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df["Close"] = df["Close"].astype(float)
    commodity_data[name] = df

# --- Helper function: find nearest previous available date within 7 days
def find_nearest_date(df, target_date, max_lag_days=7):
    available_dates = df.index[df.index <= target_date]
    if not available_dates.empty:
        nearest_date = available_dates.max()
        if (target_date - nearest_date).days <= max_lag_days:
            return nearest_date
    return None

# --- Build the feature set
feature_rows = []

LOOKBACK_DAYS = 15

for idx, row in index_compo.iterrows():
    rebalance_date = row["Rebalancing Date"]
    constituents = eval(row["Constituents (tickers)"])
    snapshot_date = rebalance_date - pd.Timedelta(days=LOOKBACK_DAYS)

    # Get Fed and ECB rates (take last known value)
    fed_latest = fed_rate[fed_rate.index <= snapshot_date]["Fed Funds Rate (%)"].iloc[-1] if not fed_rate[
                                                                fed_rate.index <= snapshot_date].empty else np.nan
    ecb_latest = ecb_rate[ecb_rate.index <= snapshot_date]["ECB Interest Rate (%)"].iloc[-1] if not ecb_rate[
                                                                ecb_rate.index <= snapshot_date].empty else np.nan

    for ticker, df in ticker_data.items():
        nearest_snapshot_date = find_nearest_date(df, snapshot_date)
        if nearest_snapshot_date is None:
            continue  # Skip if no stock data available

        close_price = df.loc[nearest_snapshot_date, "Close"]
        volume = df.loc[nearest_snapshot_date, "Volume"]
        market_cap_proxy = close_price * volume
        
        # Calculate features
        try:
            past_21d_date = nearest_snapshot_date - pd.Timedelta(days=21)
            past_5d_date = nearest_snapshot_date - pd.Timedelta(days=5)

            nearest_past_close_21d_date = find_nearest_date(df, past_21d_date)
            nearest_past_close_5d_date = find_nearest_date(df, past_5d_date)
            nearest_past_volume_5d_date = find_nearest_date(df, past_5d_date)

            if nearest_past_close_21d_date:
                past_close_21d = df.loc[nearest_past_close_21d_date, "Close"]
                return_1m = (close_price / past_close_21d) - 1
            else:
                return_1m = np.nan

            if nearest_past_close_5d_date:
                past_close_5d = df.loc[nearest_past_close_5d_date, "Close"]
                price_momentum = (close_price - past_close_5d) / past_close_5d
            else:
                price_momentum = np.nan

            if nearest_past_volume_5d_date:
                past_volume_5d = df.loc[nearest_past_volume_5d_date, "Volume"]
                volume_momentum = (volume - past_volume_5d) / past_volume_5d
            else:
                volume_momentum = np.nan

            # Rolling volatility and average volume
            past_returns = df.loc[nearest_snapshot_date - pd.Timedelta(days=30):nearest_snapshot_date, "Return"].dropna()
            past_volumes = df.loc[nearest_snapshot_date - pd.Timedelta(days=30):nearest_snapshot_date, "Volume"].dropna()

            volatility_1m = past_returns.std() if len(past_returns) > 5 else np.nan
            average_volume_1m = past_volumes.mean() if len(past_volumes) > 5 else np.nan

        except KeyError:
            # If historical data missing
            return_1m = price_momentum = volume_momentum = volatility_1m = average_volume_1m = np.nan
        
        # Get index levels
        index_features = {}
        for idx_name, idx_df in index_data.items():
            nearest_idx_date = find_nearest_date(idx_df, snapshot_date, max_lag_days=21)
            index_close = idx_df.loc[nearest_idx_date, "Close"] if nearest_idx_date else np.nan
            index_features[f"Index_{idx_name}"] = index_close

        # Get commodity levels
        for com_name, com_df in commodity_data.items():
            nearest_com_date = find_nearest_date(com_df, snapshot_date)
            com_close = com_df.loc[nearest_com_date, "Close"] if nearest_com_date else np.nan
            index_features[f"Commodity_{com_name}"] = com_close

        feature_rows.append({
            "Rebalance Date": rebalance_date,
            "Ticker": ticker,
            "Close Price": close_price,
            "Volume": volume,
            "Market Cap Proxy": market_cap_proxy,
            "Return_1m": return_1m,
            "Price_Momentum": price_momentum,
            "Volume_Momentum": volume_momentum,
            "Volatility_1m": volatility_1m,
            "Average_Volume_1m": average_volume_1m,
            "Fed Funds Rate": fed_latest,
            "ECB Interest Rate": ecb_latest,
            **index_features,
            "Label": 1 if ticker in constituents else 0
        })

# Final feature dataset
feature_df = pd.DataFrame(feature_rows)
feature_df.to_excel("FEATURES_CAC40_features_dataset.xlsx", index=False)

print("Feature generation complete.")



