import tensorflow as tf
import numpy as np


class Ia:
    def __init__(self):
        self.model = None

    def save_folder(self, name):
        self.model.save(f'Backtest/{name}.h5')

    def create_model(self):
        val_input = tf.keras.layers.Input(shape=(60, 6, 4, 1), name='input')
        x = tf.keras.layers.Conv3D(128, (3, 1, 1))(val_input)
        x = tf.keras.layers.Conv3D(128, (2, 2, 1))(x)
        x = tf.keras.layers.Conv3D(64, (4, 1, 4))(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Reshape((54, 64 * 5))(x)

        x = tf.keras.layers.LSTM(256, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(256)(x)

        x = tf.keras.layers.LSTM(128)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128)(x)

        val_output = tf.keras.layers.Dense(60)(x)

        self.model = tf.keras.models.Model(inputs=[val_input], outputs=[val_output])

    def compiler(self, loss='mean_squared_error', lr=0.001, decay=0.000_001):
        sgd = tf.keras.optimizers.SGD(lr=lr, decay=decay)
        self.model.compile(optimizer=sgd, loss=loss)

    def fit(self, datas, labels, epochs=10):
        self.model.fit(datas, labels, epochs=epochs, verbose=0)

    def predict(self, datas):
        values = self.model.predict(datas)
        return values

    def load(self, name):
        self.model = tf.keras.models.load_model(f'Backtest/{name}.h5')


if __name__ == '__main__':
    print('test')
    my_ia = Ia()
    my_ia.create_model()
    print(my_ia.model.summary())
    # my_ia.compiler()
    # my_ia.fit([np.random.random(120)], [np.random.random(30)], 100)
    # a = my_ia.predict([np.random.random(120)])
    # print(a)
    # print(type(a[0]))
