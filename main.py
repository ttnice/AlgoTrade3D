from IAObject import Ia
from DataObject import Data
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

'''
function ClickConnect(){
    console.log("Clicked on connect button"); 
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect,60000)
'''





epochs = 3
print('starting')

def main():
    my_ia = Ia()
    my_ia.create_model()
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
    history = my_ia.fit(datas, labels, epochs)
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
