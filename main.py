from IAObject import Ia
from DataObject import Data
import numpy as np
from datetime import datetime







epochs = 1

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
        now = datetime.now()
        my_ia.fit(datas, labels, 1)
        my_ia.save_folder(i)
        print(datetime.now()-now)





if __name__ == '__main__':
    main()
