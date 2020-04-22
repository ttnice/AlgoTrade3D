from IAObject import Ia
from DataObject import Data
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os

'''
function ClickConnect(){
    console.log("Clicked on connect button"); 
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect,60000)
'''

version = '5.0.4'
date = '04-20'
v = version.split('.')[0]

# Creating folder
v_path = f'Backtest/{v}/'
path = f'Backtest/{v}/{date}/'
os.makedirs(path, exist_ok=True)

initial_epoch = 6
loss = 0.0124470
epochs = 100
print('starting')
print(f'v {version} - {date}')

def main():
    my_ia = Ia(v, path, date)
    if not initial_epoch is None:
        # my_ia.load_model(v_path + 'model.json')
        my_ia.create_model()
        my_ia.load_weights(f'{path}{initial_epoch:03d}-{loss:.7f}.h5')
    else:
        my_ia.create_model()
        my_ia.save_model(v_path + 'model.json')
    my_ia.compiler()

    taille = 60
    predict_taille = 60
    my_data = Data(taille, predict_taille, 'data/scaled/Merged_2019.csv')
    # my_data = Data(taille, predict_taille, 'data/week1.csv')



    datas = []
    labels = []
    for i in range(my_data.len -taille -predict_taille):
    # for i in range(100):
        # for i in range(60):
        datas.append(list(my_data.get_data(i)))
        labels.append(list(my_data.get_predict(i)))

    datas = np.array(datas)
    labels = np.array(labels)
    start = datetime.now()
    history = my_ia.fit(datas, labels, epochs, validation_split=.1, batch_size=128, initial_epoch=initial_epoch)
    print(history.history)
    log = f'{start} - {datetime.now()} : {epochs} LOSS : {history.history["loss"][-1]}'
    # my_ia.save_folder(i)
    open('Improvement/log.txt', 'a+').write(str(log) + '\n')

    # print(i, datetime.now())
    print(history.history.keys())
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
    pass
