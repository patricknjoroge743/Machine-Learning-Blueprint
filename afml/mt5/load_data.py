"""
This script provides functionalities to download financial market data
and save it in a structured, partitioned Parquet format. Data is fetched
from MetaTrader 5 (MT5) for specified symbols and date ranges, with options
to handle existing files intelligently. It includes a user-friendly GUI prompt
to manage overwriting existing data, robust logging with 'loguru', and secure
credential management using environment variables. The script is designed to be
run as a standalone application for data acquisition, ensuring that the account
used for downloading matches the account used for loading data later.

FUntionality is provided for loading downloaded files into a DataFrame for analysis,
with options to specify columns and date ranges. The script is optimized for memory
usage and provides a clear directory structure for easy data management.

Features:
- Downloads data for a given list of symbols and a date range.
- Organizes saved data into a clear directory structure: path/symbol/year/month.parquet
- Implements a user-friendly, timed GUI prompt to ask for overwriting existing files.
- Uses 'loguru' for robust logging to both console and a file.
- Secures credentials using environment variables.
- Verifies account consistency for both downloading and loading data.
- Designed to be run as a standalone script for data acquisition.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from dask import dataframe as dd
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from afml.mt5.clean_data import clean_tick_data
from afml.util.misc import (
    date_conversion,
    is_first_weekday,
    is_last_weekday,
    log_df_info,
)

# --- Credential and Login Management ---


def get_credentials_from_env(account):
    """
    Retrieves MT5 credentials from environment variables.

    Args:
        account (str): The account name (e.g., 'MyAccount').

    Returns:
        tuple: (login, password, server) or (None, None, None) if not found.
    """
    load_dotenv()  # Load environment variables from .env file if present
    prefix = f"MT5_ACCOUNT_{account.upper()}"
    login = os.environ.get(f"{prefix}_LOGIN")
    password = os.environ.get(f"{prefix}_PASSWORD")
    server = os.environ.get(f"{prefix}_SERVER")

    if not all([login, password, server]):
        logger.error(f"Missing one or more environment variables for account '{account}'.")
        logger.error(f"Please set {prefix}_LOGIN, {prefix}_PASSWORD, and {prefix}_SERVER.")
        return None, None, None

    if login.isnumeric():
        login = int(login)

    return login, password, server


def login_mt5(account, timeout=60000, verbose=True):
    """
    Logs in to a MetaTrader5 account using credentials from environment variables.

    Args:
        account (str): Account name to log in to.
        timeout (int): Connection timeout in milliseconds.
        verbose (bool): Whether to print detailed connection information.

    Returns:
        str: The account name if login is successful, otherwise None.
    """
    import MetaTrader5 as mt5

    logger.info(f"Attempting to log in to MT5 with account: {account}")
    login, password, server = get_credentials_from_env(account)

    if not login:
        return None

    if not mt5.initialize(login=login, password=password, server=server, timeout=timeout):
        logger.error(f"MT5 initialize() failed for account {account}. Error: {mt5.last_error()}")
        mt5.shutdown()
        return

    logger.success(f"Successfully logged in to MT5 as {account}.")
    if verbose:
        logger.info(f"MT5 Version: {mt5.version()}")
        terminal_info = mt5.terminal_info()
        if terminal_info:
            logger.info(f"Connected to {terminal_info.name} at {terminal_info.path}")
        else:
            logger.warning("Could not retrieve terminal info.")

    return account


# --- Data Validation and Verification ---


def verify_or_create_account_info(data_path, current_account_name):
    """
    Checks if the data directory is associated with the correct account.
    If no account info exists, it creates it.

    Args:
        data_path (Path): The root path of the data directory.
        current_account_name (str): The name of the account currently in use.

    Returns:
        bool: True if the account is verified, False otherwise.
    """
    account_info_file = data_path / "account_info.json"
    current_account_name = current_account_name.upper()

    if account_info_file.exists():
        try:
            with open(account_info_file, "r") as f:
                stored_info = json.load(f)
                stored_name = stored_info.get("account_name")

            if stored_name and stored_name != current_account_name:
                logger.error(
                    f"Account Mismatch! This directory ('{data_path.name}') is for account '{stored_name}'."
                )
                logger.error(
                    f"Current operation is for account '{current_account_name}'. Aborting to prevent data errors."
                )
                return False
            elif not stored_name:
                # File exists but is malformed, so we fix it.
                logger.warning("Account info file is malformed. Overwriting with current account.")
                with open(account_info_file, "w") as f:
                    json.dump({"account_name": current_account_name}, f, indent=4)

        except json.JSONDecodeError:
            logger.warning(
                f"Could not read account info file. Overwriting with current account: '{current_account_name}'."
            )
            with open(account_info_file, "w") as f:
                json.dump({"account_name": current_account_name}, f, indent=4)
    else:
        logger.info(
            f"First time use for this directory. Associating it with account '{current_account_name}'."
        )
        with open(account_info_file, "w") as f:
            json.dump({"account_name": current_account_name}, f, indent=4)

    return True


# --- Data Fetching and Saving ---


def get_ticks(symbol, start_date, end_date, datetime_index=True, verbose=True):
    """
    Downloads tick data from the MT5 terminal for a given period.

    Args:
        symbol (str): The financial instrument symbol (e.g., 'EURUSD').
        start_date (pd.Timestamp): The timezone-aware start date for the data range.
        end_date (pd.Timestamp): The timezone-aware end date for the data range.
        datetime_index (bool): Set 'time' column to DatetimeIndex.

    Returns:
        pd.DataFrame: A DataFrame containing the tick data, or an empty DataFrame if
                      no data is found or an error occurs.
    """
    import MetaTrader5 as mt5

    if not mt5.terminal_info():  # Check if connection is still active
        logger.error("MT5 connection lost. Cannot download data.")
        return pd.DataFrame()

    try:
        start_date, end_date = date_conversion(start_date, end_date)
        # mt5.symbol_select(symbol, True)
        ticks = mt5.copy_ticks_range(symbol, start_date, end_date, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            logger.warning(
                f"No tick data returned for {symbol} from {start_date.date()} to {end_date.date()}."
            )
            return pd.DataFrame()

        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
        df.drop(columns=["time_msc"], inplace=True)

        if datetime_index:
            df.set_index("time", inplace=True)

        # Keep only columns with meaningful data
        df = df.loc[:, df.any()]

        # Optimize memory usage
        for col in ["bid", "ask"]:
            if col in df.columns:
                df[col] = df[col].astype("float32")

        if verbose:
            log_df_info(df)

        return df

    except Exception as e:
        logger.error(f"An error occurred while getting ticks for {symbol}: {e}")
        return pd.DataFrame()


def get_bars(symbol, timeframe, start_date, end_date, datetime_index=True, verbose=True):
    """
    Downloads bar (OHLCV) data from the MT5 terminal for a given period.

    Args:
        symbol (str): The financial instrument symbol (e.g., 'EURUSD').
        timeframe (int): MT5 timeframe constant (e.g., mt5.TIMEFRAME_M1, mt5.TIMEFRAME_H1).
        start_date (pd.Timestamp): Timezone-aware start date.
        end_date (pd.Timestamp): Timezone-aware end date.
        datetime_index (bool): Set 'time' column to DatetimeIndex.
        verbose (bool): Print DataFrame info.

    Returns:
        pd.DataFrame: A DataFrame containing OHLCV data, or empty if no data/error.
    """
    import MetaTrader5 as mt5

    if not mt5.terminal_info():
        logger.error("MT5 connection lost. Cannot download data.")
        return pd.DataFrame()

    try:
        start_date, end_date = date_conversion(start_date, end_date)
        timeframe = getattr(mt5, f"TIMEFRAME_{timeframe}")
        mt5.symbol_select(symbol, True)
        bars = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if bars is None or len(bars) == 0:
            logger.warning(
                f"No bar data returned for {symbol} from {start_date.date()} to {end_date.date()}."
            )
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        if datetime_index:
            df.set_index("time", inplace=True)

        # Optimize memory usage
        for col in ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]:
            if col in df.columns:
                df[col] = df[col].astype("float32")

        if verbose:
            log_df_info(df)

        return df

    except Exception as e:
        logger.error(f"An error occurred while getting bars for {symbol}: {e}")
        return pd.DataFrame()


def process_symbol(symbol, start_dt, end_dt, data_path, account_name):
    """Worker function to download data for a single symbol."""
    try:
        login_mt5(account_name)  # Each worker needs its own login
    except Exception as e:
        return {symbol: f"login_failed: {e}"}

    symbol_path = data_path / symbol
    dates_from = pd.date_range(start=start_dt, end=end_dt, freq="MS", tz="UTC")
    dates_to = pd.date_range(start=start_dt, end=end_dt, freq="ME", tz="UTC") + pd.Timedelta(days=1)

    missing_data = []

    for start, end in zip(dates_from, dates_to):
        year_path = symbol_path / str(start.year)
        year_path.mkdir(parents=True, exist_ok=True)

        file = year_path / f"month-{start.month:02d}.parquet"
        log_msg_prefix = f"{symbol}  -> Month {start.strftime('%Y-%m')}..."

        if file.exists():
            df0 = pd.read_parquet(file)
            if not df0.empty:
                first, start = [x.date() for x in df0.index[[0, -1]]]
                if is_first_weekday(first) and is_last_weekday(start):
                    logger.info(f"{log_msg_prefix} Exists—Skipping download")
                    continue
                else:
                    logger.info(f"{log_msg_prefix} Exists—Appending from {start} to {end}")
        else:
            df0 = pd.DataFrame()

        df1 = get_ticks(symbol, start, end, verbose=False)
        df = pd.concat([df0, df1])

        if not df.empty:
            df = clean_tick_data(df)
            df.to_parquet(file, engine="pyarrow", compression="zstd")
            logger.success(f"{log_msg_prefix} Saved {len(df):,} rows")
        else:
            logger.warning(f"{log_msg_prefix} No data found")
            missing_data.append(start.strftime("%Y-%m"))
            try:
                year_path.rmdir()
            except:
                pass

    mt5.shutdown()  # Each worker should shutdown its own connection
    return {symbol: missing_data}


def save_data_to_parquet(symbols, start_date, end_date, account_name, path=None):
    """
    Downloads and saves tick data to a partitioned Parquet structure.

    Args:
        symbols (Union[str, list, tuple]): A single symbol or a collection of symbols to download.
        start_date (Union[str, dt, pd.Timestamp]): The start date for the data range.
        end_date (Union[str, dt, pd.Timestamp]): The end date for the data range.
        account_name (str): The name of the account used for the download.
        path (Union[str, Path]): The root folder where data will be saved.
    Returns:
        None
    """

    data_path = Path(path) if path is not None else Path().home() / "tick_data_parquet"
    data_path.mkdir(parents=True, exist_ok=True)

    date_range = date_conversion(start_date, end_date)
    if not date_range:
        return
    start_dt, end_dt = date_range

    if isinstance(symbols, str):
        symbols = [symbols]

    missing_data = {}

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_symbol, symbol, start_dt, end_dt, data_path, account_name
            ): symbol
            for symbol in symbols
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading symbols"):
            symbol = futures[future]
            try:
                result = future.result()
                missing_data.update(result)
            except Exception as e:
                logger.critical(f"Worker for {symbol} failed: {e}")

    # Summary logging
    logger.info("Download process finished.")
    if missing_data and any(missing_data.values()):
        logger.warning("Missing data summary:")
        for symbol, months in missing_data.items():
            logger.warning(f"  - {symbol}: {', '.join(months)}")

    logger.success(f"All operations complete. Files saved to {data_path}")


# --- Loading Data from Files ---


def load_tick_data(
    symbol,
    start_date,
    end_date,
    account_name,
    path=None,
    columns=None,
    verbose=True,
):
    """
    Loads tick data from a partitioned Parquet structure after verifying account.

    Args:
        path (Union[str, Path]): The root folder where the data is stored.
        symbol (str): The financial instrument symbol to load.
        start_date (Union[str, dt, pd.Timestamp]): The start date of the desired data range.
        end_date (Union[str, dt, pd.Timestamp]): The end date of the desired data range.
        account_name (str): The account name to verify against the data directory.
        columns (Optional[list]): A list of specific columns to load. Loads all if None.
        verbose (bool): If True, logs detailed DataFrame info upon successful load.

    Returns:
        pd.DataFrame: A DataFrame with the requested tick data, or an empty DataFrame
                      if the account verification fails, dates are invalid, or an error occurs.
    """
    try:
        root_path = Path(path)
    except TypeError:
        root_path = Path.home() / "tick_data_parquet"

    if not verify_or_create_account_info(root_path, account_name):
        return pd.DataFrame()

    date_range = date_conversion(start_date, end_date)
    if date_range:
        start_dt, end_dt = date_range
        fname = root_path / symbol.upper()
    else:
        return pd.DataFrame()

    try:
        filters = [("time", ">=", start_dt), ("time", "<=", end_dt)]

        ddf = dd.read_parquet(
            fname,
            columns=columns,
            filters=filters,
            engine="pyarrow",
        )
        df = ddf.compute()

        to_drop = []
        for col in df.columns:
            # Drop columns
            if any(np.isnan(df[col].unique())):
                to_drop.append(col)
            # Optimise dtype of flags column for memory
            if col == "flags":
                mem = df.memory_usage(deep=True).sum()  # memory before downcasting
                dtype_orig = df["flags"].dtype
                limit = df["flags"].max()
                for x in (8, 16, 32):
                    dtype = f"uint{x}"
                    if dtype_orig != dtype and np.iinfo(dtype).max >= limit:
                        df = df.astype({"flags": dtype})
                        mem = (mem - df.memory_usage(deep=True).sum()) / 1024**2
                        logger.info(
                            f"Converted flags from {dtype_orig} to {df['flags'].dtype} saving {mem:,.1f} MB"
                        )
                        break

        if to_drop:
            df.drop(columns=to_drop, inplace=True)
            logger.info(f"Dropped empty columns {to_drop}")

        if not df.index.is_monotonic_increasing:
            df.sort_index(inplace=True)

        logger.success(f"Loaded {len(df):,} rows of {symbol} tick data for account {account_name}")
        if verbose:
            log_df_info(df)

        return df

    except Exception as e:
        logger.error(f"Failed to load data for {symbol}. Error: {e}")
        return pd.DataFrame()


def get_tick_data_in_memory(
    symbols, start_date, end_date, account_name, clean_data=True, verbose=True
):
    """
    Fetches tick data and returns concatenated DataFrame without saving to disk.
    Perfect for feeding into bar creation pipelines.

    Args:
        symbols (Union[str, list]): Symbol or list of symbols to fetch
        start_date (Union[str, datetime]): Start date for data range
        end_date (Union[str, datetime]): End date for data range
        account_name (str): MT5 account name for login
        clean_data (bool): Whether to apply cleaning to tick data
        verbose (bool): Whether to log progress details

    Returns:
        pd.DataFrame: Concatenated tick data for all symbols
    """
    if isinstance(symbols, str):
        symbols = [symbols]

    all_tick_data = []

    for symbol in tqdm(symbols, desc="Fetching symbols in memory"):
        try:
            # Login for each symbol (handles connection pooling)
            logged_in = login_mt5(account_name, verbose=False)
            if not logged_in:
                logger.error(f"Failed to login for {symbol}")
                continue

            # Get raw tick data
            tick_df = get_ticks(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                datetime_index=True,
                verbose=False,
            )

            if tick_df.empty:
                logger.warning(f"No data retrieved for {symbol}")
                continue

            # Apply cleaning if requested
            if clean_data:
                tick_df = clean_tick_data(tick_df)
                if tick_df is None or tick_df.empty:
                    logger.warning(f"Data cleaning removed all data for {symbol}")
                    continue

            tick_df["symbol"] = symbol  # Add symbol identifier
            all_tick_data.append(tick_df)

            if verbose:
                logger.success(f"Fetched {len(tick_df):,} ticks for {symbol}")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue
        finally:
            mt5.shutdown()  # Cleanup connection

    if not all_tick_data:
        logger.error("No data was successfully fetched for any symbol")
        return pd.DataFrame()

    # Concatenate all symbol data
    combined_df = pd.concat(all_tick_data, axis=0)
    combined_df.sort_index(inplace=True)

    logger.success(f"Combined {len(combined_df):,} ticks across {len(all_tick_data)} symbols")

    if verbose:
        log_df_info(combined_df)

    return combined_df


# Complete data pipeline
def create_multi_timeframe_bars(symbols, start_date, end_date, account_name, bar_configs=None):
    """
    End-to-end pipeline: Fetch ticks → Create multiple bar types
    """
    if bar_configs is None:
        bar_configs = [
            {"bar_type": "time", "bar_size": "H1", "price": "mid_price"},
            {"bar_type": "tick", "bar_size": 1000, "price": "bid_ask"},
            {"bar_type": "volume", "bar_size": 1000000, "price": "mid_price"},
        ]

    # 1. Fetch all tick data in memory
    tick_data = get_tick_data_in_memory(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        account_name=account_name,
        clean_data=True,
        verbose=True,
    )

    if tick_data.empty:
        logger.error("No tick data available for bar creation")
        return {}

    # 2. Create different bar types
    bars_dict = {}
    for config in bar_configs:
        bar_type = config["bar_type"]
        bar_size = config["bar_size"]

        logger.info(f"Creating {bar_type} bars with size {bar_size}")

        # Group by symbol and create bars for each
        symbol_bars = []
        for symbol in symbols:
            symbol_ticks = tick_data[tick_data["symbol"] == symbol]
            if symbol_ticks.empty:
                continue

            try:
                bars_df = make_bars(
                    tick_df=symbol_ticks,
                    bar_type=bar_type,
                    bar_size=bar_size,
                    price=config["price"],
                    verbose=True,
                )
                bars_df["symbol"] = symbol
                symbol_bars.append(bars_df)
            except Exception as e:
                logger.error(f"Failed creating {bar_type} bars for {symbol}: {e}")

        if symbol_bars:
            bars_dict[f"{bar_type}_{bar_size}"] = pd.concat(symbol_bars, axis=0)

    return bars_dict


# --- Main Execution Block ---
if __name__ == "__main__":
    MAJORS = [
        "EURUSD",
        "USDJPY",
        "GBPUSD",
        "USDCHF",
        "AUDUSD",
        "USDCAD",
        "NZDUSD",
        "XAUUSD",
    ]

    CRYPTO = [
        "ADAUSD",
        "BTCUSD",
        "DOGUSD",
        "ETHUSD",
        "LNKUSD",
        "LTCUSD",
        "XLMUSD",
        "XMRUSD",
        "XRPUSD",
    ]

    # --- 1. User Configuration ---
    CONFIG = {
        "save_path": Path.home() / "tick_data_parquet",
        "symbols_to_download": CRYPTO,
        "account_to_use": "FundedNext_STLR2_6K",  # This name MUST match the one used in your environment variables
        "start_date": "2022-01-01",
        "end_date": "2024-12-31",
        "verbose_login": True,
    }

    # --- 2. Setup Logging ---
    # Configure logger to output to console and a file for persistent records.
    log_path = CONFIG["save_path"] / "data_download.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # The default logger is console-only. Add a file sink.
    logger.add(
        log_path,
        rotation="10 MB",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    logger.info("--- Starting New Data Download Session ---")

    # --- 3. Login to MT5 ---
    logged_in_account = login_mt5(account=CONFIG["account_to_use"], verbose=CONFIG["verbose_login"])

    # --- 4. Run Downloader ---
    if logged_in_account:
        save_data_to_parquet(
            symbols=CONFIG["symbols_to_download"],
            start_date=CONFIG["start_date"],
            end_date=CONFIG["end_date"],
            account_name=logged_in_account,
            path=CONFIG["save_path"],
        )

        # --- 6. Shutdown MT5 Connection ---
        mt5.shutdown()
        logger.info("--- MT5 Connection Closed. Session End ---")

    else:
        logger.critical("Could not log in to MetaTrader 5. Aborting all operations.")
