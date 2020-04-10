from IAObject import Ia
from DataObject import Data
import numpy as np
from datetime import datetime

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
    my_ia.compiler()

    taille = 60
    predict_taille = 60
    my_data = Data(taille, predict_taille, 'data/week1.csv')

    datas = []
    labels = []
    for i in range(my_data.len -taille -predict_taille):
        datas.append(list(my_data.get_data(i)))
        labels.append(list(my_data.get_predict(i)))

    datas = np.array(datas)
    labels = np.array(labels)
    for i in range(epochs):
        my_ia.fit(datas, labels, 500)
        my_ia.save_folder(i)
        print(i, datetime.now())





if __name__ == '__main__':
    main()
