# Imports
import pandas as pd

# Functions
def load_data(path, base_path = r'/content/drive/MyDrive/Modeling_Project/Data/'):
    """
    Loads data from base path based on sub-path (path) in base path.
    """
    return pd.read_csv(base_path + path)


def clean_hourly_data(df_10min, column_names, agg_dict=None):
    """
    Clean 10-minute data and aggregate to hourly, keeping only complete hours.

    Parameters:
    -----------
    df_10min : DataFrame
        Ten-minute data with 'time' column
    column_names : list
        List of column names of the dataframe
    agg_dict : dict, optional
        Aggregation dictionary. If None, auto-aggregates all numeric columns with 'mean'

    Returns:
    --------
    DataFrame : Clean hourly data
    """

    df = df_10min.copy()
    # Convert Hebrew columns to English
    df.columns = column_names
    # Strip whitespace from 'time'
    df['time'] = df['time'].str.strip()

    # Convert 'time' to datetime
    df['time'] = pd.to_datetime(
    df['time'],
    format='%d/%m/%Y %H:%M',
    errors='coerce'
)
    df['hour'] = df['time'].dt.floor('h')

    # Get columns to check for missing data (exclude 'station', 'time', and 'hour')
    data_cols = [col for col in df.columns if col not in ['station', 'hour', 'time']]

    # Convert data_cols
    for col in data_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.'),
            errors='coerce'
        )

    # Filter to winter months
    df = df[df['time'].dt.month.isin([10,11,12,1,2,3])]

    # Find valid hours: exactly 6 readings AND no NaN values
    valid_hours = (
        df.groupby('hour')
        .filter(lambda x: (len(x) == 6) and (~x[data_cols].isna().any().any()))
        ['hour'].unique()
    )

    # If no agg_dict provided, use 'mean' for all numeric columns except rain_amount which uses 'sum'
    if agg_dict is None:
        agg_dict = {
                col: ('sum' if col == 'rain_amount' else 'mean')
                for col in data_cols
            }


    # Aggregate valid hours
    df_hourly = (
        df[df['hour'].isin(valid_hours)]
        .groupby('hour')
        .agg(agg_dict)
        .reset_index()
        .rename(columns={'hour': 'time'})
    )

    return df_hourly

def align_dataframes(df1, df2, dt, print_summary=True):
    """
    Align two dataframes based on time lag, keeping only matching pairs.

    For each row at time t in df1, keeps it only if df2 has data at t+dt.
    Both dataframes must already be cleaned (no missing data).

    Parameters:
    -----------
    df1 : DataFrame
        First dataframe with 'time' column (e.g., JLM)
    df2 : DataFrame
        Second dataframe with 'time' column (e.g., BD)
    dt : int
        Time lag in hours (df1 at time t, df2 at time t+dt)
    print_summary : bool
        Print alignment summary (default: True)

    Returns:
    --------
    tuple : (df1_aligned, df2_aligned)
        Aligned dataframes with matching time pairs
    """

    df1 = df1.copy()
    df2 = df2.copy()
    df1['time'] = pd.to_datetime(df1['time'])
    df2['time'] = pd.to_datetime(df2['time'])

    # Create time-shifted version of df2
    df2['time_shifted'] = df2['time'] - pd.Timedelta(hours=dt)

    # Find matching times
    merged = df1.merge(
        df2[['time_shifted']],
        left_on='time',
        right_on='time_shifted',
        how='inner'
    )

    valid_df1_times = merged['time'].values
    valid_df2_times = (merged['time'] + pd.Timedelta(hours=dt)).values

    # Filter both dataframes
    df1_aligned = df1[df1['time'].isin(valid_df1_times)].reset_index(drop=True)
    df2_aligned = df2[df2['time'].isin(valid_df2_times)].drop(columns='time_shifted').reset_index(drop=True)

    if print_summary:
        print(f"df1: {len(df1)} → {len(df1_aligned)} rows")
        print(f"df2: {len(df2)} → {len(df2_aligned)} rows")
        print(f"Matched pairs: {len(df1_aligned)}")

    return df1_aligned, df2_aligned