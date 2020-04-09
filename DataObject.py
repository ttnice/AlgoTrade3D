import pandas as pd
import numpy as np


class Data:
    def __init__(self, taille, predict_taille, path):
        self.df = pd.read_csv(path)
        self.len = len(self.df)
        self.order = [
            'CHFJPY-Open', 'CHFJPY-High', 'CHFJPY-Low', 'CHFJPY-Close',
            'EURCHF-Open', 'EURCHF-High', 'EURCHF-Low', 'EURCHF-Close',
            'EURJPY-Open', 'EURJPY-High', 'EURJPY-Low', 'EURJPY-Close',
            'EURUSD-Open', 'EURUSD-High', 'EURUSD-Low', 'EURUSD-Close',
            'USDCHF-Open', 'USDCHF-High', 'USDCHF-Low', 'USDCHF-Close',
            'USDCAD-Open', 'USDCAD-High', 'USDCAD-Low', 'USDCAD-Close'
        ]
        self._df = self.df[self.order]
        self.predict_df = self.df['EURCHF-Close']
        self.taille = taille
        self.predict_taille = predict_taille

    def get_data(self, idx):
        # get the DataFrame from the _df DataFrame
        data = self._df.loc[idx:idx + self.taille - 1]

        # get values from the data into numpy array
        data = data.values

        # reshapping values to a 3D tensors with (Timedata, Devise, OHLC)
        data = data.reshape((self.taille, 6, 4, 1))

        return data

    def get_predict(self, idx):
        # get the prediction Series from the predict_df Series
        predict = self.predict_df.loc[idx + self.taille:idx + self.taille + self.predict_taille - 1]
        # get the prediction values from the prediction Series
        predict = predict.values
        return predict

if __name__ == '__main__':
    taille = 60
    predict_taille = 60
    my_data = Data(taille, predict_taille, 'data/week1.csv')

    datas = []
    labels = []
    for i in range(my_data.len - taille - predict_taille):
        datas.append(my_data.get_data(i))
        labels.append(my_data.get_predict(i))

    datas = np.array(datas)
    predict = np.array(labels)