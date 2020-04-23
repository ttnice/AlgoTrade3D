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

version = '7.0.0'
date = '01'
v = version.split('.')[0]

# Creating folder
v_path = f'Backtest/{v}/'
path = f'Backtest/{v}/{date}/'
os.makedirs(path, exist_ok=True)

initial_epoch = None
loss = 0.0124470
epochs = 20
print('starting')
print(f'v {version} - {date}')


def create_model():
    my_ia = Ia(v, path, date)
    val_input = tf.keras.layers.Input(shape=(60, 1), name='input')

    x = tf.keras.layers.Dropout(0.2)(val_input)
    x = tf.keras.layers.Dense(512, activation='linear')(x)

    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation='linear')(x)

    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation='linear')(x)

    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation='linear')(x)

    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='linear')(x)

    val_output = tf.keras.layers.Dense(60, activation='linear')(x)

    my_ia.model = tf.keras.models.Model(inputs=[val_input], outputs=[val_output])
    my_ia.save_model(v_path+ f'{date}/' + 'model.json')







def main():
    taille = 60
    predict_taille = 60
    my_data = Data(taille, predict_taille, 'data/scaled/Merged_2019.csv')
    # my_data = Data(taille, predict_taille, 'data/week1.csv')

    datas = []
    labels = []
    for i in range(my_data.len -taille -predict_taille):
        # datas.append(list(my_data.get_data(i)))
        datas.append(list(my_data.get_predict(i-taille)))
        labels.append(list(my_data.get_predict(i)))

    datas = np.array(datas)
    labels = np.array(labels)

    for i in range(1, 2):
        date = '%02d'%i
        path = f'Backtest/{v}/{date}/'
        my_ia = Ia(v, path, date)
        my_ia.load_model(v_path + f'{date}/' + 'model.json')
        # my_ia.load_weights(f'{path}{initial_epoch:03d}-{loss:.7f}.h5')
        my_ia.compiler()
        my_ia.callbacks_list(checkpoint=False)

        my_ia.fit(datas, labels, epochs, validation_split=.1, batch_size=128, initial_epoch=initial_epoch)
        my_ia.save(v_path + f'{date}/' + 'model.h5')
        my_ia.save_weights(v_path + f'{date}/' + 'model_weights.h5')


if __name__ == '__main__':
    # create_model()
    main()
    pass
