from IAObject import Ia
from DataObject import Data
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os
import tensorflow as tf

'''
tensorboard --logdir Backtest/logs/
'''

'''
function ClickConnect(){
    console.log("Clicked on connect button"); 
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect,60000)
'''
'''
x = tf.keras.layers.Reshape((60, 24))(x)

x = tf.keras.layers.LSTM(512, return_sequences=True, activation='linear')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(512, activation='linear')(x)

x = tf.keras.layers.LSTM(512, return_sequences=True, activation='linear')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(512, activation='linear')(x)

x = tf.keras.layers.LSTM(256, return_sequences=True, activation='linear')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, activation='linear')(x)

x = tf.keras.layers.LSTM(256, return_sequences=True, activation='linear')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, activation='linear')(x)

x = tf.keras.layers.LSTM(128, activation='linear')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128, activation='linear')(x)

val_output = tf.keras.layers.Dense(60, activation='linear')(x)'''


def create_model(start):
    # Creating folder
    date = '%02d' % start
    v_path = f'Backtest/{v}/'
    path = f'Backtest/{v}/{date}/'
    os.makedirs(path, exist_ok=True)

    my_ia = Ia(v, path, date)

    val_input = tf.keras.layers.Input(shape=(60,), name='input')

    x = tf.keras.layers.Dense(128, activation='linear')(val_input)

    x = tf.keras.layers.Dense(128, activation='linear')(x)
    x = tf.keras.layers.Dense(128, activation='linear')(x)
    x = tf.keras.layers.Dense(128, activation='linear')(x)
    x = tf.keras.layers.Dense(128, activation='linear')(x)
    x = tf.keras.layers.Dense(128, activation='linear')(x)
    x = tf.keras.layers.Dense(128, activation='linear')(x)
    x = tf.keras.layers.Dense(128, activation='linear')(x)
    x = tf.keras.layers.Dense(128, activation='linear')(x)
    x = tf.keras.layers.Dense(128, activation='linear')(x)
    x = tf.keras.layers.Dense(128, activation='linear')(x)
    x = tf.keras.layers.Dense(128, activation='linear')(x)
    x = tf.keras.layers.Dense(128, activation='linear')(x)

    val_output = tf.keras.layers.Dense(60, activation='linear')(x)

    my_ia.model = tf.keras.models.Model(inputs=[val_input], outputs=[val_output])
    my_ia.save_model(v_path+ f'{date}/' + 'model.json')


def main(start, end, epochs, load):
    target_time = 7_200
    taille = 60
    predict_taille = 60

    datas = []
    labels = []

    '''
    # my_data = Data(taille, predict_taille, 'data/scaled/Merged_2019.csv')
    my_data = Data(taille, predict_taille, 'data/week1.csv')
    for i in range(my_data.len -taille -predict_taille):
        # datas.append(list(my_data.get_data(i)))
        datas.append(list(my_data.get_predict(i-taille)))
        labels.append(list(my_data.get_predict(i)))
    '''
    for i in range(target_time -taille -predict_taille):
        delta = (np.random.random() + 0.1) * 10 ** -4
        value = np.random.random()
        datas.append([value+(delta*i) for i in range(taille)])
        labels.append([value+(delta*(i+taille)) for i in range(predict_taille)])

    datas = np.array(datas)
    labels = np.array(labels)
    print(datas.shape)
    print(labels.shape)
    for i in range(start, end):
        date = '%02d'%i
        print(date)
        path = f'Backtest/{v}/{date}/'
        my_ia = Ia(v, path, date)
        my_ia.load_model(v_path + f'{date}/' + 'model.json')
        if load:
            my_ia.load_weights(f'{path}model_weights.h5')
        my_ia.compiler(loss='mean_squared_error', lr=0.01, decay=0.000_001, momentum=0.9, nesterov=True)
        my_ia.init_callbacks(checkpoint=False)

        my_ia.fit(datas, labels, epochs, validation_split=.1, batch_size=128, initial_epoch=load)
        my_ia.save(v_path + f'{date}/' + 'model.h5')
        my_ia.save_weights(v_path + f'{date}/' + 'model_weights.h5')
        print(f"Finished  {date}")

def predict(start, end):
    target_time = 7_200
    taille = 60
    predict_taille = 60

    datas = []
    labels = []

    '''
    # my_data = Data(taille, predict_taille, 'data/scaled/Merged_2019.csv')
    my_data = Data(taille, predict_taille, 'data/week1.csv')
    for i in range(my_data.len -taille -predict_taille):
        # datas.append(list(my_data.get_data(i)))
        datas.append(list(my_data.get_predict(i-taille)))
        labels.append(list(my_data.get_predict(i)))
    '''

    '''
    value = np.random.random()
    datas.append([value for i in range(taille)])
    labels.append([value for i in range(predict_taille)])'''
    delta = (np.random.random()+0.1)*10**-4
    value = np.random.random()
    datas.append([value+(delta*i) for i in range(taille)])
    labels.append([value+(delta*(i+taille)) for i in range(predict_taille)])

    datas = np.array(datas)
    labels = np.array(labels)
    print(datas.shape)
    print(labels.shape)
    plt.plot(np.concatenate((datas[0], labels[0])), label="Data")
    for i in range(start, end):
        date = '%02d'%i
        print(date)
        path = f'Backtest/{v}/{date}/'
        my_ia = Ia(v, path, date)
        my_ia.load_model(v_path + f'{date}/' + 'model.json')
        my_ia.load_weights(f'{path}model_weights.h5')

        prediction = my_ia.predict(datas)
        print(f"Finished  {date}")
        plt.plot(np.concatenate((datas[0], prediction[0])), label=f"{date}")
    delta = 10**-6
    # plt.ylim((value-delta, value+delta))
    plt.legend()
    plt.show()

version = '8.0.0'
v = version.split('.')[0]
v_path = f'Backtest/{v}/'


start = 7
n = 1
epochs = 1
if __name__ == '__main__':
    create_model(start)
    main(start, start+n, epochs, 00)
    # predict(start, start+n)
    pass
