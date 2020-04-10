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





epochs = 50
print('starting')

def main():
    my_ia = Ia()
    my_ia.create_model()
    my_ia.load_weights('16')

    taille = 60
    predict_taille = 60
    my_data = Data(taille, predict_taille, 'data/week1.csv')

    datas = []
    labels = []
    # for i in range(my_data.len -taille -predict_taille):
    for i in range(10):
        idx = random.randint(0, my_data.len -taille -predict_taille)
        datas.append(list(my_data.get_data(idx)))
        labels.append(list(my_data.get_predict(idx)))

    labels = np.array(labels)
    predicts = my_ia.predict(datas)
    for i in range(len(datas)):
        plt.plot(labels[i], label="Label")
        plt.plot(predicts[i], label="Predict")

        plt.legend()
        plt.show(block=True)





if __name__ == '__main__':
    main()
