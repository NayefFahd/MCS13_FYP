# Pickle and Keras for model loading
from pickle import load
from keras.models import load_model
import numpy as np


class Forecast:
    def __init__(self) -> None:

        # Import Models
        self.model_weekly4 = load_model('weekly_dataset_model4.h5')
        self.model_daily7 = load_model('daily_dataset_model7.h5')
        self.model_daily60 = load_model('daily_dataset_model60.h5')
        self.model_1hour = load_model('1h_dataset_model6.h5')
        self.model_30min = load_model('30min_dataset_model14.h5')
        self.model_15min = load_model('15min_dataset_model30.h5')
        self.model_5min = load_model('5min_dataset_model60.h5')

        # Import Scalars
        self.weeklyScaler = load(open('scaler_weekly.sav', 'rb'))
        self.dailyScaler = load(open('scaler_daily.sav', 'rb'))
        self.hourlyScaler = load(open('scaler_1h.sav', 'rb'))
        self.scaler_30 = load(open('scaler_30min.sav', 'rb'))
        self.scaler_15 = load(open('scaler_15min.sav', 'rb'))
        self.scaler_5 = load(open('scaler_5min.sav', 'rb'))

    def predict_next_week(self, df):
        df.dropna(inplace=True)
        # Normalize data
        scaled_data = self.weeklyScaler.fit_transform(df)
        # training_data_len = 4
        x_train = []
        y_train = []
        training_data_len = int(len(df)*0.9)

        #Create Training Dataset
        train_data = scaled_data[0:training_data_len, : ]
        for i in range(4,len(train_data)):
            x_train.append(train_data[i-4:i,0])
            y_train.append(train_data[i,0])
        
        #Convert x_train and y_train to numpy array
        x_train , y_train = np.array(x_train), np.array(y_train)
        # Reshape data
        x_train  = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

        self.model_weekly4.fit(x_train, y_train, epochs = 10, batch_size = 1,verbose=False)
        self.model_weekly4.save('weekly_dataset_model4.h5')
        self.model_weekly4 = load_model('weekly_dataset_model4.h5')

        # Create the testing dataset
        test_data = scaled_data[training_data_len-4:, :]

        # Create data sets x_test and y_test
        x_test = []
        y_test = scaled_data[training_data_len:, :]

        for j in range(4, len(test_data)):
            x_test.append(test_data[j-4:j, 0])

        new_x_test = x_test[-1]
        new_x_test = np.delete(x_test[-1], 0)
        new_x_test = np.append(new_x_test, y_test[-1])

        # reshape data
        new_x_test = np.reshape(new_x_test, (1, 4, 1))

        # get the models predicted price values
        predictions = self.model_weekly4.predict(new_x_test)
        predictions = self.weeklyScaler.inverse_transform(predictions)
        return predictions[0][0]

    def predict_next_day(self, df):
        df.dropna(inplace=True)
        # Normalize data
        scaled_data = self.dailyScaler.fit_transform(df)
        x_train = []
        y_train = []
        training_data_len = int(len(df)*0.9)

        #Create Training Dataset
        train_data = scaled_data[0:training_data_len, : ]
        for i in range(60,len(train_data)):
            x_train.append(train_data[i-60:i,0])
            y_train.append(train_data[i,0])
        
        #Convert x_train and y_train to numpy array
        x_train , y_train = np.array(x_train), np.array(y_train)
        # Reshape data
        x_train  = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

        self.model_daily60.fit(x_train, y_train, epochs = 10, batch_size = 1,verbose=False)
        self.model_daily60.save('daily_dataset_model60.h5')
        self.model_daily60 = load_model('daily_dataset_model60.h5')

        # Create the testing dataset
        test_data = scaled_data[training_data_len-60:, :]

        # Create data sets x_test and y_test
        x_test = []
        y_test = scaled_data[training_data_len:, :]

        for j in range(60, len(test_data)):
            x_test.append(test_data[j-60:j, 0])

        new_x_test = x_test[-1]
        new_x_test = np.delete(x_test[-1], 0)
        new_x_test = np.append(new_x_test, y_test[-1])

        # reshape data
        new_x_test = np.reshape(new_x_test, (1, 60, 1))

        # get the models predicted price values
        predictions = self.model_daily60.predict(new_x_test)
        predictions = self.dailyScaler.inverse_transform(predictions)
        return predictions[0][0]

    def predict_next_hour(self, df):
        df.dropna(inplace=True)
        # Normalize data
        scaled_data = self.hourlyScaler.fit_transform(df)
        x_train = []
        y_train = []
        training_data_len = int(len(df)*0.9)

        #Create Training Dataset
        train_data = scaled_data[0:training_data_len, : ]
        for i in range(6,len(train_data)):
            x_train.append(train_data[i-6:i,0])
            y_train.append(train_data[i,0])
        
        #Convert x_train and y_train to numpy array
        x_train , y_train = np.array(x_train), np.array(y_train)
        # Reshape data
        x_train  = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

        self.model_1hour.fit(x_train, y_train, epochs = 10, batch_size = 1,verbose=False)
        self.model_1hour.save('1h_dataset_model6.h5')
        self.model_1hour = load_model('1h_dataset_model6.h5')

        # Create the testing dataset
        test_data = scaled_data[training_data_len-6:, :]

        # Create data sets x_test and y_test
        x_test = []
        y_test = scaled_data[training_data_len:, :]

        for j in range(6, len(test_data)):
            x_test.append(test_data[j-6:j, 0])

        new_x_test = x_test[-1]
        new_x_test = np.delete(x_test[-1], 0)
        new_x_test = np.append(new_x_test, y_test[-1])

        # reshape data
        new_x_test = np.reshape(new_x_test, (1, 6, 1))

        # get the models predicted price values
        predictions = self.model_1hour.predict(new_x_test)
        predictions = self.hourlyScaler.inverse_transform(predictions)
        return predictions[0][0]

    def predict_next_30m(self, df):
        df.dropna(inplace=True)
        # Normalize data
        scaled_data = self.scaler_30.fit_transform(df)
        x_train = []
        y_train = []
        training_data_len = int(len(df)*0.9)

        #Create Training Dataset
        train_data = scaled_data[0:training_data_len, : ]
        for i in range(14,len(train_data)):
            x_train.append(train_data[i-14:i,0])
            y_train.append(train_data[i,0])
        
        #Convert x_train and y_train to numpy array
        x_train , y_train = np.array(x_train), np.array(y_train)
        # Reshape data
        x_train  = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

        self.model_30min.fit(x_train, y_train, epochs = 10, batch_size = 1,verbose=False)
        self.model_30min.save('30min_dataset_model14.h5')
        self.model_30min = load_model('30min_dataset_model14.h5')

        # Create the testing dataset
        test_data = scaled_data[training_data_len-14:, :]

        # Create data sets x_test and y_test
        x_test = []
        y_test = scaled_data[training_data_len:, :]

        for j in range(14, len(test_data)):
            x_test.append(test_data[j-14:j, 0])

        new_x_test = x_test[-1]
        new_x_test = np.delete(x_test[-1], 0)
        new_x_test = np.append(new_x_test, y_test[-1])

        # reshape data
        new_x_test = np.reshape(new_x_test, (1, 14, 1))

        # get the models predicted price values
        predictions = self.model_30min.predict(new_x_test)
        predictions = self.scaler_30.inverse_transform(predictions)
        return predictions[0][0]
