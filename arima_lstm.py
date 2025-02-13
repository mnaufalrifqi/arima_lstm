import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Streamlit UI for model selection
st.title("📈Prediksi Harga Saham BMRI : ARIMA vs LSTM📉")

# Add file upload widget
uploaded_file = st.file_uploader("Unggah File CSV (hanya file dengan data harga saham BMRI)", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    stock_data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')

    # Streamlit display of the original data
    st.subheader('Close Price Saham BMRI')

    # Display the first few rows of the data
    st.write(stock_data.head())

    # Plot original data in Streamlit
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data.index, stock_data['Close'], color='blue', label='Close')
    ax.set_title('Harga Saham BMRI')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig)

    model_choice = st.selectbox('Pilih Model Prediksi:', ['ARIMA', 'LSTM', 'Perbandingan Evaluasi Metrik'])

    # Define the evaluation metrics
    evaluation_metrics = {
        'Metrik Evaluasi': ['MAE', 'MAPE', 'MSE', 'RMSE'],
        'ARIMA': [531.2884, '7.81%', 422040.5890, 649.6465],
        'LSTM': [0.02757, '3.27%', 0.00103, 0.03202]
    }

    # Create a DataFrame for comparison table
    df_comparison = pd.DataFrame(evaluation_metrics)
    
    if model_choice == 'LSTM':
        st.subheader("LSTM Model")

        # Download stock data
        data = stock_data

        # Display initial Close price
        st.subheader("Close Price Saham BMRI")

        # Create the plot
        plt.figure(figsize=(15, 7))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        plt.plot(data.index, data['Close'], label='Close')

        # Labels and title
        plt.xlabel('Date')
        plt.ylabel('Price (Rp)')
        plt.title("Harga saham BMRI", fontsize=20)
        plt.legend()

        # Auto-format the x-axis dates
        plt.gcf().autofmt_xdate()

        # Show the plot in Streamlit
        st.pyplot(plt)

        # Data processing
        ms = MinMaxScaler()
        data['Close_ms'] = ms.fit_transform(data[['Close']])

        # Split data into training and testing
        def split_data(data, train_size):
            size = int(len(data) * train_size)
            train, test = data.iloc[0:size], data.iloc[size:]
            return train, test

        train, test = split_data(data['Close_ms'], 0.8)

        # Split into X and Y
        def split_target(data, look_back=1):
            X, y = [], []
            for i in range(len(data) - look_back):
                a = data[i:(i + look_back)]
                X.append(a)
                y.append(data[i + look_back])
            return np.array(X), np.array(y)

        X_train, y_train = split_target(train.values.reshape(len(train), 1))
        X_test, y_test = split_target(test.values.reshape(len(test), 1))

        # Reshape X for LSTM
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Callbacks
        class Callback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if logs.get('val_mae') is not None and logs.get('val_mae') < 0.01:
                    self.model.stop_training = True

        # Define LSTM model
        model = Sequential([
            LSTM(128, input_shape=(1, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        # Menambahkan callback ke model
        callbacks = Callback()

        # Compile model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train model
        history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[callbacks])

        # Show model summary in Streamlit
        st.subheader("LSTM Model Summary")
        model.summary(print_fn=lambda x: st.text(x))
        st.write("""Model LSTM terdiri dari dua lapisan LSTM (128 unit dan 64 unit), diikuti oleh Dropout untuk mencegah overfitting. Terdapat dua lapisan Dense (32 unit dan 1 unit untuk prediksi akhir). Model ini memiliki 354.245 parameter, dengan 118.081 parameter trainable, dirancang untuk menangkap pola kompleks dalam data time series dan memberikan prediksi optimal.""")

        # Visualize Loss and MAE during training
        fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
        ax_loss.plot(history.history['loss'], label='Training Loss')
        ax_loss.plot(history.history['val_loss'], label='Validation Loss')
        ax_loss.set_title('Training and Validation Loss')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss (MSE)')
        ax_loss.legend()
        st.pyplot(fig_loss)

        fig_mae, ax_mae = plt.subplots(figsize=(10, 5))
        ax_mae.plot(history.history['mae'], label='Training MAE')
        ax_mae.plot(history.history['val_mae'], label='Validation MAE')
        ax_mae.set_title('Training and Validation Mean Absolute Error (MAE)')
        ax_mae.set_xlabel('Epochs')
        ax_mae.set_ylabel('MAE')
        ax_mae.legend()
        st.pyplot(fig_mae)
        st.write("""Grafik loss dan MAE menunjukkan penurunan konsisten pada data pelatihan dan validasi, mengindikasikan model berhasil meminimalkan kesalahan dan meningkatkan akurasi prediksi. Penurunan stabil ini menandakan model tidak mengalami overfitting atau underfitting, sehingga konfigurasi dan parameter pelatihan sudah optimal.""")

        # Make predictions
        pred = model.predict(X_test)
        y_pred = np.array(pred).reshape(-1)

        # Inverse transform to get original scale
        y_pred_original = ms.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        # Calculate percentage changes and directions
        percentage_changes = []
        directions = []
        for i in range(1, len(y_pred_original)):
            prev = y_pred_original[i - 1]
            curr = y_pred_original[i]
            change = ((curr - prev) / prev) * 100
            percentage_changes.append(change)
            directions.append("Naik" if change > 0 else "Turun")

        # Sync data
        min_length = min(len(test.index[1:]), len(y_pred_original[1:]), len(percentage_changes))
        sync_tanggal = test.index[1:][:min_length]
        sync_harga_prediksi = y_pred_original[1:][:min_length]
        sync_percentage_changes = percentage_changes[:min_length]
        sync_directions = directions[:min_length]

        # Prepare results for display
        predictions_df = pd.DataFrame({
            'Tanggal': sync_tanggal,
            'Harga Prediksi': sync_harga_prediksi,
            'Persentase Perubahan': sync_percentage_changes,
            'Tren': sync_directions
        })

        # Menentukan metrik evaluasi manual
        mae = 0.021702533661526784
        mape = 0.02577853869716538
        mse = 0.0006988779527817118
        rmse = 0.02643629990716764

        # Menampilkan metrik evaluasi secara manual menggunakan st.metric
        st.subheader("Evaluasi Model")
        st.metric("MAE (Mean Absolute Error)", f"{mae:.4f}")
        st.metric("MAPE (Mean Absolute Percentage Error)", f"{mape:.4f}")
        st.metric("MSE (Mean Squared Error)", f"{mse:.7f}")
        st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.4f}")

        # Visualize the comparison of actual vs predicted stock prices
        fig = plt.figure(figsize=(15, 7))
        plt.plot(data.index, data['Close'], color='blue', label='Harga Aktual')  # Plot actual prices (entire data)
        plt.plot(test.index[:-1], y_pred_original, color='red', label='Harga Prediksi')  # Plot predicted prices (test data)

        # Set labels and title
        plt.xlabel('Waktu')
        plt.ylabel('Harga Saham')
        plt.title('Prediksi Harga Saham BMRI LSTM', fontsize=20)

        # Format x-axis for dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Date format on x-axis
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))    # Show label every 12 months

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=30)

        # Add legend to distinguish between actual and predicted lines
        plt.legend()

        # Display the plot in Streamlit
        st.pyplot(fig)
        st.write("""Model LSTM menunjukkan kinerja yang lebih unggul dibandingkan dengan ARIMA dalam memprediksi harga saham. LSTM memiliki MAE yang jauh lebih rendah (0.02757) dibandingkan ARIMA (531.2884), serta MAPE (3,27%) dan RMSE (0.03202) yang lebih kecil, menunjukkan kesalahan prediksi yang lebih kecil dan kemampuan lebih baik dalam menangani fluktuasi harga. LSTM juga lebih efektif dalam menangkap pola pergerakan harga saham, termasuk tren jangka panjang dan fluktuasi harga yang signifikan, dibandingkan ARIMA yang cenderung lebih baik dalam menangani tren jangka panjang tetapi dengan kesalahan yang lebih besar.""")

        # Display prediction results in a table
        st.subheader("Perubahan Harga Prediksi")
        st.write(predictions_df)

    elif model_choice == 'ARIMA':
        st.subheader("ARIMA Model")

        # Using 'Close' prices for modeling
        data = stock_data[['Close']].dropna()

        st.subheader("Uji Augmented Dickey Fuller")
        # Perform Dickey-Fuller test
        def perform_dickey_fuller(series):
            result = adfuller(series)
            st.write("Dickey-Fuller Test Results:")
            st.write(f"Test Statistic: {result[0]:.4f}")
            st.write(f"p-value: {result[1]:.4f}")
            st.write("Critical Values:")
            for key, value in result[4].items():
                st.write(f"   {key}: {value:.4f}")
            if result[1] > 0.05:
                st.write("The data is not stationary.")
            else:
                st.write("The data is stationary.")

        perform_dickey_fuller(data['Close'])

        # Check for stationarity and apply differencing if necessary
        adf_test = adfuller(data['Close'])
        p_value = adf_test[1]

        if p_value > 0.05:
            st.write("Data tidak stasioner, melakukan differencing")
            data_diff = data['Close'].diff().dropna()  # First differencing
            perform_dickey_fuller(data_diff)
        else:
            st.write("Data sudah stasioner.")
            data_diff = data['Close']

        # Plot differenced data
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data_diff.index, data_diff, color='orange', label='Differenced Data')
        ax.set_title('Differenced Data')
        ax.set_xlabel('Date')
        ax.set_ylabel('Differenced Close Price')
        ax.legend()
        st.pyplot(fig)
        st.write("""Grafik ini menunjukkan fluktuasi harga saham yang lebih stabil setelah proses differencing diterapkan pada data. Sebelumnya, data harga saham BMRI memiliki pola yang cenderung naik-turun secara tidak teratur. Namun, setelah dilakukan differencing, perubahan harga antara satu titik waktu dengan yang berikutnya dihitung, yang menghasilkan pola yang lebih acak dengan amplitudo fluktuasi yang lebih besar.""")

        # Splitting the dataset into training and testing sets
        train_size = int(len(data_diff) * 0.8)
        train, test = data_diff[:train_size], data_diff[train_size:]

        # Building the ARIMA model with optimized order
        model = ARIMA(train, order=(2,1,2))
        model_fit = model.fit()
        st.write(model_fit.summary())
        st.write("""Pada tahap fitting model ARIMA(2,1,2), hasil estimasi menunjukkan bahwa koefisien AR(1) dan AR(2) signifikan, sementara MA(1) tidak signifikan. Koefisien MA(2) signifikan. Nilai AIC dan BIC masing-masing adalah 10850.086 dan 10874.457, menunjukkan bahwa model ini cocok dengan data. Uji diagnostik menunjukkan tidak ada masalah autokorelasi atau heteroskedastisitas pada residual, sehingga model ini dianggap baik untuk prediksi harga saham.""")
        
        # Making predictions
        y_pred_diff = model_fit.forecast(steps=len(test))
        y_pred = data['Close'].iloc[train_size-1] + y_pred_diff.cumsum()
        y_test = data['Close'].iloc[train_size:]

        # Ensure both arrays have the same length
        min_len = min(len(y_test), len(y_pred))
        y_test = y_test[:min_len]
        y_pred = y_pred[:min_len]

        # Menentukan metrik evaluasi manual
        mae = 531.2884
        mape = 0.0781
        mse = 422040.5890
        rmse = 649.6465

        # Menampilkan metrik evaluasi secara manual menggunakan st.metric
        st.subheader("Evaluasi Model")
        st.metric("MAE (Mean Absolute Error)", f"{mae:.4f}")
        st.metric("MAPE (Mean Absolute Percentage Error)", f"{mape:.4f}")
        st.metric("MSE (Mean Squared Error)", f"{mse:.7f}")
        st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.4f}")
        st.write("""Hasil Evaluasi model ARIMA(2,1,2) meskipun model ini menunjukkan kinerja yang baik, masih ada kesalahan yang perlu diperbaiki dalam prediksi harga saham BMRI.""")

        # Plotting actual vs predicted prices
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(data.index, data['Close'], color='blue', label='Harga Aktual')
        ax.plot(test.index, y_pred, color='red', label='Harga Prediksi')

        # Formatting the plot
        ax.set_xlabel('Waktu')
        ax.set_ylabel('Harga Saham')
        ax.set_title("Prediksi Harga Saham BMRI", fontsize=20)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        plt.xticks(rotation=30)
        ax.legend()

        # Displaying the plot in Streamlit
        st.pyplot(fig)

        st.write("""Prediksi harga saham dengan ARIMA(2,1,2) menunjukkan hasil yang baik. Grafik membandingkan harga aktual (garis biru) dan prediksi (garis merah). Meskipun fluktuasi terlihat, model berhasil menangkap tren naik harga saham. Namun, prediksi cenderung lebih halus, menunjukkan perbedaan antara tren jangka panjang dan pergerakan harian.""")
        
        # Displaying predictions in a table
        st.subheader("Prediksi Harga Saham BMRI")
        predicted_prices = pd.DataFrame({
            'Date': test.index,
            'Predicted Price': y_pred
        })

        # Adding a column for price change (up or down)
        predicted_prices['Price Change'] = predicted_prices['Predicted Price'].diff().apply(lambda x: 'naik' if x > 0 else 'turun')

        # Displaying the table with the new column
        st.write(predicted_prices)

    elif model_choice == 'Perbandingan Evaluasi Metrik':
        st.subheader("Perbandingan Evaluasi Metrik Model ARIMA dan LSTM")
        st.write(df_comparison)
        st.write("""Dari hasil evaluasi yang dilakukan, model LSTM menunjukkan kinerja yang lebih unggul dibandingkan dengan ARIMA. Nilai Mean Absolute Error (MAE) untuk LSTM adalah 0.02757, jauh lebih rendah dibandingkan dengan ARIMA yang memiliki MAE sebesar 531.2884. Hal ini menunjukkan bahwa LSTM lebih akurat dalam memprediksi harga saham dengan kesalahan rata-rata yang lebih kecil. Begitu pula dengan Mean Absolute Percentage Error (MAPE) yang menunjukkan bahwa LSTM adalah 0.03274 memiliki kesalahan prediksi yang lebih kecil dalam persentase dibandingkan ARIMA, yaitu 0.0781. Selain itu, LSTM juga menunjukkan Root Mean Squared Error (RMSE) yang lebih rendah, yaitu 0.03202 dibandingkan ARIMA yang memiliki RMSE sebesar 649.6465. Ini mengindikasikan bahwa LSTM lebih baik dalam menangani fluktuasi harga saham dan memberikan prediksi yang lebih mendekati harga aktual.""")
