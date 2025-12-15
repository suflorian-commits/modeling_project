# Imports
import numpy as np
import pandas as pd

# Functions
def forecast_rain_binary(
    source_rh_t,          # RH at source at time t [%]
    source_temp_t,        # Temperature at source at time t [°C]
    source_press_t,       # Pressure at source at time t [hPa]
    dest_press_t,         # Pressure at destination at time t [hPa]
    wind_speed,           # Wind speed at source at time t [m/s]
    wind_direction,       # Wind direction at source at time t [degrees]
    gamma=0.75,           # Critical RH threshold (tunable parameter)
    optimal_wind_dir=90,  # Optimal wind direction [degrees] (tunable parameter)
    min_wind_speed=1.0    # Minimum effective wind speed [m/s] (tunable parameter)
):
    """
    Binary rain forecast at destination based on current source conditions.

    Returns:
        rain_forecast: 1 if rain expected (>1mm at the moment), 0 otherwise
        rh_lifted: RH after orographic lifting [%]
        lag_hours: Expected time for air mass to reach destination [hours]
        u_effective: Effective wind speed component toward destination [m/s]

    Notes:
        - Wind direction assumed to be meteorological convention (direction FROM)
          TO DO: Verify this assumption
        - optimal_wind_dir and min_wind_speed are tunable parameters
    """

    # Constants
    DELTA_Z = 735        # Elevation difference [m] from Source (Beit Dagan) to Goal (Jerusalem)
    GAMMA_D = 0.0098     # Dry adiabatic lapse rate [K/m]
    DISTANCE = 45000     # Distance between stations [m] (45km)

    # Step 1: Calculate wind alignment
    # Wind direction is meteorological (FROM), so if wind is FROM west (270°),
    # it's blowing eastward toward destination
    # TO DO: VERIFY wind direction convention!!

    wind_alignment = np.cos(np.radians(wind_direction - optimal_wind_dir))
    u_effective = wind_speed * max(0, wind_alignment)

    # Step 2: Check if wind is favorable for moisture transport
    if u_effective < min_wind_speed:
        # Wind not blowing from source to destination, or too weak
        return 0, 0.0, np.inf, u_effective

    # Step 3: Calculate transport lag (hours until air mass reaches destination)
    lag_hours = DISTANCE / (u_effective * 3600)

    # Step 4: Calculate saturation vapor pressure at source (from Tetens formula)
    es_source = 6.112 * np.exp(17.67 * source_temp_t / (source_temp_t + 243.5))

    # Step 5: Calculate saturation specific humidity at source
    qs_source = 0.622 * es_source / (source_press_t - 0.622 * es_source)

    # Step 6: Calculate actual specific humidity at source from RH
    q_source = (source_rh_t / 100.0) * qs_source

    # Step 7: Account for adiabatic cooling during orographic uplift
    temp_after_lift = source_temp_t - GAMMA_D * DELTA_Z

    # Step 8: Calculate saturation conditions at destination elevation
    es_dest_lifted = 6.112 * np.exp(17.67 * temp_after_lift / (temp_after_lift + 243.5))
    qs_dest_lifted = 0.622 * es_dest_lifted / (dest_press_t - 0.622 * es_dest_lifted)

    # Step 9: Calculate RH after orographic lifting
    rh_lifted = (q_source / qs_dest_lifted) * 100

    # Step 10: Binary rain decision based on critical RH threshold
    # Rain occurs (>1mm) if RH after lifting exceeds critical threshold
    if rh_lifted > gamma * 100:
        rain_forecast = 1
    else:
        rain_forecast = 0

    return rain_forecast, rh_lifted, lag_hours, u_effective


def apply_forecast_to_timeseries(
    source_data,          # DataFrame with source station data
    dest_data,            # DataFrame with destination station data
    h_ahead,              # Forecast horizon [hours]
    gamma=0.75,
    optimal_wind_dir=90,
    min_wind_speed=1.0
):
    """
    Apply the forecast model to time series data.

    Given conditions at source at time t, forecast rain at destination at time t+h_ahead.

    Args:
        source_data: DataFrame with columns ['timestamp', 'rh', 'temp', 'pressure',
                                              'wind_speed', 'wind_direction']
        dest_data: DataFrame with columns ['timestamp', 'rain_amount', 'pressure']
        h_ahead: How many hours ahead to forecast

    Returns:
        DataFrame with columns ['timestamp', 'rain_forecast', 'rain_actual',
                                'rh_lifted', 'lag_hours', 'u_effective']
    """


    results = []

    for idx, source_row in source_data.iterrows():
        t_source = source_row['timestamp']
        t_dest = t_source + pd.Timedelta(hours=h_ahead)

        # Find corresponding destination data
        dest_row = dest_data[dest_data['timestamp'] == t_dest]

        if dest_row.empty:
            continue

        # Get destination pressure at forecast time
        dest_press = dest_row['pressure'].values[0]

        # Make forecast
        rain_forecast, rh_lifted, lag_hours, u_effective = forecast_rain_binary(
            source_rh_t=source_row['rh'],
            source_temp_t=source_row['temp'],
            source_press_t=source_row['pressure'],
            dest_press_t=dest_press,
            wind_speed=source_row['wind_speed'],
            wind_direction=source_row['wind_direction'],
            gamma=gamma,
            optimal_wind_dir=optimal_wind_dir,
            min_wind_speed=min_wind_speed
        )

        # Get actual rain (binary: 1 if >1mm, 0 otherwise)
        rain_actual = 1 if dest_row['rain_amount'].values[0] > 1.0 else 0

        results.append({
            'timestamp_source': t_source,
            'timestamp_dest': t_dest,
            'rain_forecast': rain_forecast,
            'rain_actual': rain_actual,
            'rh_lifted': rh_lifted,
            'lag_hours': lag_hours,
            'u_effective': u_effective
        })

    return pd.DataFrame(results)