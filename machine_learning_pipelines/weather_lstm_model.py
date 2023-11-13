# Import Modules
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import load_model

def emissions_lstm_forecast(hist_df, forecast_df, series_to_forecast_df, target_column):
    """
    Train an LSTM on historical data, test on 30% of it and forecast the next 31 days.

    :param hist_df: DataFrame of historical weather data.
    :param forecast_df: DataFrame of next 31 days of forecasted weather data.
    :param series_to_forecast_df: DataFrame of historical series data (e.g., customer load or system demand).
    :return: Series of forecasted values for the next 31 days.
    """
    
    # Combine historical weather data with the series to be forecasted
    combined_df = pd.merge(hist_df, series_to_forecast_df, on=['day'], how='inner')
    
    # Splitting the data: 70% for training and 30% for testing
    split_idx = int(0.7 * combined_df.shape[0])
    train_df = combined_df.iloc[:split_idx]
    test_df = combined_df.iloc[split_idx:]

    # 1. Normalize the combined data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_normalized = scaler.fit_transform(train_df.drop(columns=['day']))
    train_normalized_df = pd.DataFrame(train_normalized, columns=train_df.columns[1:])
    test_normalized = scaler.transform(test_df.drop(columns=['day']))
    test_normalized_df = pd.DataFrame(test_normalized, columns=test_df.columns[1:])

    # 2. Prepare LSTM sequences
    features = list(set(train_df.columns) - {'day', target_column})
    X_train, y_train = prepare_data(train_normalized_df, features=features, target=target_column, time_steps=TIME_STEPS)
    X_test, y_test = prepare_data(test_normalized_df, features=features, target=target_column, time_steps=TIME_STEPS)

    # 3. Build and train the LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate the model on test data
    y_pred_test = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    print(f"Mean Absolute Error on Test Data: {mae_test}")
    print(f"Root Mean Squared Error on Test Data: {rmse_test}")

    # 4. Normalize and prepare the forecast data for predictions
    forecast_normalized = scaler.transform(forecast_df.drop(columns=['day']))
    forecast_normalized_df = pd.DataFrame(forecast_normalized, columns=forecast_df.columns[1:])
    X_forecast, _ = prepare_data(forecast_normalized_df, features=features, target=target_column, time_steps=TIME_STEPS)
    
    # 5. Make predictions for the next 31 days
    forecasted_values = model.predict(X_forecast)
    forecasted_values_original_scale = forecasted_values * (scaler.data_max_[train_df.columns.get_loc(target_column)] - scaler.data_min_[train_df.columns.get_loc(target_column)]) + scaler.data_min_[train_df.columns.get_loc(target_column)]

    return pd.Series(forecasted_values_original_scale.flatten())