from IAObject import Ia
from DataObject import Data
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import random

'''
function ClickConnect(){
    console.log("Clicked on connect button"); 
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect,60000)
'''


version = '5.0.2'
date = '04-20'
v = version.split('.')[0]

# Creating folder
v_path = f'Backtest/{v}/'
path = f'Backtest/{v}/{date}/'

# model name
initial_epoch = 6
loss = 0.0124470
print('starting')

def main():
    my_ia = Ia(version, path, date)
    # my_ia.load_model(v_path + 'model.json')
    my_ia.create_model()
    my_ia.load_weights(f'{path}{initial_epoch:03d}-{loss:.7f}.h5')

    # my_ia2 = Ia(version, path, date)
    # my_ia2.create_model()
    # my_ia2.load_model('Backtest/4/model.json')
    # my_ia2.load_weights('Backtest/4/04-20/033-0.0103890.h5')

    taille = 60
    predict_taille = 60
    my_data = Data(taille, predict_taille, 'data/scaled/Merged_2019.csv')

    datas = []
    labels = []
    # for i in range(my_data.len -taille -predict_taille):
    for i in range(my_data.len -taille -predict_taille-10, my_data.len -taille -predict_taille):
        idx = random.randint(0, my_data.len -taille -predict_taille)
        datas.append(list(my_data.get_data(idx)))
        labels.append(list(my_data.get_predict(idx)))

    labels = np.array(labels)
    predicts = my_ia.predict(datas)
    # predicts2 = my_ia2.predict(datas)
    for i in range(len(datas)):
        plt.plot(labels[i], label="Label")
        plt.plot(predicts[i], label="Predict")
        # plt.plot(predicts2[i], label="Predict2")
        # plt.ylim((0, 1))

        plt.legend()
        plt.show(block=True)





if __name__ == '__main__':
    main()
