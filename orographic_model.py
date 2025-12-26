import pandas as pd
import numpy as np
def run_orographic_model(source_df, dest_df, h_ahead=3, gamma=0.75):
    """
    Vectorized implementation of the binary rain model.
    
    Parameters:
    -----------
    source_df : pd.DataFrame
        Must contain: 'timestamp', 'rh', 'temp', 'pressure', 'wind_speed', 'wind_direction'
    dest_df : pd.DataFrame
        Must contain: 'timestamp', 'rain_amount', 'pressure'
    h_ahead : int
        Forecast horizon in hours (default 3 hours based on typical lag BD->JLM)
    gamma : float
        RH threshold (0.75 = 75%)
    
    Returns:
    --------
    results_df : pd.DataFrame
    """
    
    # 1. Align Data based on Forecast Horizon (h_ahead)
    # We want to match Source(t) with Destination(t + h_ahead)
    
    # Create a shifted destination timestamp to match source
    dest_df_shifted = dest_df.copy()
    dest_df_shifted['timestamp_match'] = dest_df_shifted['timestamp'] - pd.Timedelta(hours=h_ahead)
    
    # Merge on timestamp (Inner join: we only keep times where we have both data)
    merged_df = pd.merge(
        source_df, 
        dest_df_shifted[['timestamp_match', 'rain_amount', 'pressure']], 
        left_on='timestamp', 
        right_on='timestamp_match', 
        suffixes=('_src', '_dest')
    )
    
    # Rename for clarity
    merged_df = merged_df.rename(columns={'pressure_dest': 'pressure_at_target_time'})
    
    # --- PHYSICAL MODEL VECTORIZED CALCULATIONS ---
    
    # Constants
    DELTA_Z = 735.0        # Elevation difference [m]
    GAMMA_D = 0.0098       # Dry adiabatic lapse rate [K/m]
    DISTANCE = 45000.0     # Distance [m]
    OPTIMAL_DIR = 270.0    # FROM West (270 degrees)
    
    # 1. Wind Alignment
    # Wind comes FROM 270. We want the component towards East (90).
    # Ideally: cos(wind_dir - 270). 
    # If wind is 270: cos(0) = 1. If wind is 90: cos(-180) = -1.
    wind_alignment = np.cos(np.radians(merged_df['wind_direction'] - OPTIMAL_DIR))
    
    # Clip negative values (wind blowing away from Jerusalem doesn't help)
    wind_alignment = np.maximum(0, wind_alignment)
    
    u_effective = merged_df['wind_speed'] * wind_alignment
    
    # 2. Calculate Transport Lag (Physical travel time)
    # Avoid division by zero
    lag_hours = np.where(u_effective > 0.1, 
                         DISTANCE / (u_effective * 3600), 
                         999.0) # If wind is 0, lag is infinite
    
    # 3. Saturation Vapor Pressure at Source (Tetens)
    t_src = merged_df['temp']
    es_source = 6.112 * np.exp(17.67 * t_src / (t_src + 243.5))
    
    # 4. Specific Humidity at Source (q_source)
    p_src = merged_df['pressure']
    qs_source = 0.622 * es_source / (p_src - 0.622 * es_source)
    q_source = (merged_df['rh'] / 100.0) * qs_source
    
    # 5. Adiabatic Cooling (Lifting)
    temp_lifted = t_src - (GAMMA_D * DELTA_Z)
    
    # 6. Saturation at Destination (using Lifted Temp + Destination Pressure)
    p_dest = merged_df['pressure_at_target_time']
    es_dest_lifted = 6.112 * np.exp(17.67 * temp_lifted / (temp_lifted + 243.5))
    qs_dest_lifted = 0.622 * es_dest_lifted / (p_dest - 0.622 * es_dest_lifted)
    
    # 7. Final RH Calculation
    rh_lifted = (q_source / qs_dest_lifted) * 100.0
    
    # 8. Binary Decision
    rain_forecast = (rh_lifted > (gamma * 100)).astype(int)
    
    # Filter by wind: If wind is too weak (< 1.0 m/s), force forecast to 0
    # (The moisture technically lifts, but doesn't travel fast enough)
    rain_forecast = np.where(u_effective < 1.0, 0, rain_forecast)
    
    # 9. Actual Rain Binary (> 1.0 mm is significant rain)
    rain_actual = (merged_df['rain_amount'] > 1.0).astype(int)
    
    # --- ASSEMBLE RESULTS ---
    results_df = pd.DataFrame({
        'timestamp_source': merged_df['timestamp'],
        'timestamp_dest': merged_df['timestamp'] + pd.Timedelta(hours=h_ahead),
        'rain_forecast': rain_forecast,
        'rain_actual': rain_actual,
        'rh_lifted': rh_lifted,
        'u_effective': u_effective,
        'lag_hours': lag_hours,
        'wind_speed': merged_df['wind_speed'],
        'wind_direction': merged_df['wind_direction']
    })
    
    return results_df
