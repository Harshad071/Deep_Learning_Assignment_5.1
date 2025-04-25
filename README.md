# 🚀 Stock Market Prediction using LSTM & BiLSTM

This repository contains the implementation of LSTM and BiLSTM-based models for predicting future stock prices using historical stock market data. The project leverages deep learning techniques to forecast stock prices, applying **Long Short-Term Memory (LSTM)** and **Bidirectional LSTM (BiLSTM)** networks for time series prediction.

## 📖 Overview

The main objective of this project is to predict future stock prices based on historical stock data. The models compare the standard LSTM and BiLSTM architectures and evaluate them using RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) metrics.

Key steps in the project include:

- Data preprocessing and normalization.
- Model training using LSTM and BiLSTM.
- Performance evaluation with RMSE and MAE.
- Visualization of actual vs predicted stock prices.

## 🗃️ Dataset

**Source**: [NSE India Historical Index Data](https://www.nseindia.com/reports-indices-historical-index-data)

**Key Attributes**:
- Date
- Open Price
- High Price
- Low Price
- Close Price
- Volume
- Adjusted Close Price

For this project, we focus on forecasting the **Close Price** for a particular stock over time.

## 🛠️ Methodology

### Data Exploration & Preprocessing:
1. **Load and inspect the dataset**: Examine the dataset structure, including missing values.
2. **Filter relevant columns**: Extract only the necessary features for forecasting, specifically focusing on **Close Price**.
3. **Date processing**: Convert the date fields to `datetime` objects and sort the data.
4. **Handle missing values**: Fill or drop missing values based on the dataset's condition.

### Normalization:
- Normalize the **Close Price** data using **MinMaxScaler** to scale the values between 0 and 1 for better model performance.

### Sequence Generation:
- Use a **sliding window approach** (e.g., 60 days) to create sequences that capture temporal dependencies in the stock price data.

### Model Development:
1. **Build a Standard LSTM Model**: Train an LSTM model with dropout regularization and the Adam optimizer, using Mean Squared Error as the loss function.
2. **Build a Bidirectional LSTM (BiLSTM) Model**: Create a BiLSTM model that processes the data in both forward and backward directions to capture more temporal dependencies.
3. **Hyperparameters**: Carefully tune the hyperparameters to optimize model performance.

### Training and Evaluation:
1. **Data Split**: Split the dataset into **training (80%)** and **testing (20%)** sets.
2. **Model Training**: Train both models using **EarlyStopping** and **ModelCheckpoint** callbacks to prevent overfitting and save the best model.
3. **Evaluation**: Evaluate model performance on the test set using **RMSE** and **MAE** metrics.

### Visualization:
- Plot **actual vs predicted stock prices** for both LSTM and BiLSTM models to visually compare their performance.

## 📊 Visualizations

The project includes the following visualizations:
- **Raw Time Series Plot**: A visualization of historical stock prices.
- **Scaled Data Plot**: A check to ensure proper normalization of stock price data.
- **Prediction vs Actual Plots**: Separate plots for both **LSTM** and **BiLSTM** models, showing actual vs predicted stock prices over the testing period.

## 📈 Our Model Performance

The models were evaluated on the test set with the following metrics:

- **Standard LSTM**:
  - **Test RMSE**: 630.21
  - **Test MAE**: 525.58

- **Bidirectional LSTM**:
  - **Test RMSE**: 582.40
  - **Test MAE**: 498.84

These metrics suggest that the models are capable of effectively predicting stock prices with a reasonable margin of error.

## 🔍 Insights

### Methodological Similarity:
- Both the LSTM and BiLSTM models are designed to learn from temporal data sequences, emphasizing preprocessing, normalization, and sequence generation. These steps are key in time series forecasting.

### Performance Metrics:
- Our results, including RMSE and MAE, demonstrate that **BiLSTM** outperforms the standard LSTM model, although both models show promise in capturing stock price trends.

### Future Work:
- **Hyperparameter tuning**: Further optimization of model parameters could improve accuracy.
- **Feature enrichment**: Integrating additional features such as market news, trading volumes, or technical indicators could enhance the models' forecasting capabilities.
- **Ensemble methods**: Combining multiple models could improve prediction accuracy.

## 🎯 Conclusion

This project successfully demonstrates the use of **LSTM-based models** for **stock market prediction**. By employing both **Standard LSTM** and **BiLSTM**, we observe that BiLSTM captures more temporal patterns due to its bidirectional processing. The results validate the potential of deep learning in stock price forecasting.
## 📚 References

1. **Stock Market Prediction Using LSTM Recurrent Neural Network**
   - **Published in**: *ScienceDirect*
   - **Summary**: This paper explores the application of LSTM networks for stock market prediction, focusing on the precision of machine learning algorithms and the impact of training epochs on model performance.
   - **Link**: [Stock Market Prediction Using LSTM Recurrent Neural Network](https://www.sciencedirect.com/science/article/pii/S1877050920304865)



