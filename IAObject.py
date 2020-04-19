import tensorflow as tf
import numpy as np


class Ia:
    def __init__(self, v, path):
        self.model = None
        self.v = v
        self.path = path

    def create_model(self):
        val_input = tf.keras.layers.Input(shape=(60, 6, 4, 1), name='input')
        x = tf.keras.layers.Conv3D(128, (3, 1, 1), activation='linear')(val_input)
        x = tf.keras.layers.Conv3D(128, (2, 2, 1), activation='linear')(x)
        x = tf.keras.layers.Conv3D(64, (4, 1, 4), activation='linear')(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Reshape((54, 64 * 5))(x)

        x = tf.keras.layers.LSTM(256, return_sequences=True, activation='linear')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(256, activation='linear')(x)

        x = tf.keras.layers.LSTM(128, activation='linear')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='linear')(x)

        val_output = tf.keras.layers.Dense(60, activation='linear')(x)

        self.model = tf.keras.models.Model(inputs=[val_input], outputs=[val_output])

    def compiler(self, loss='mean_squared_error', lr=0.001, decay=0.000_1):
        sgd = tf.keras.optimizers.SGD(lr=lr, decay=decay)
        self.model.compile(optimizer=sgd, loss=loss)

    def fit(self, datas, labels, epochs=10):
        filepath = f'{self.path}'+"{epoch:03d}-{loss:.7f}.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        return self.model.fit(datas, labels, batch_size=128, validation_split=.2, epochs=epochs, verbose=1, callbacks=callbacks_list, shuffle=True)

    def predict(self, datas):
        values = self.model.predict(datas)
        return values

    def save(self, name):
        self.model.save(name)

    def save_weights(self, name):
        self.model.save_weights(name)

    def save_model(self, name):
        model_json = self.model.to_json()
        with open(name, "w") as json_file:
            json_file.write(model_json)
            json_file.close()

    def load(self, name):
        self.model = tf.keras.models.load_model(name)

    def load_weights(self, name):
        self.model.load_weights(name)

    def load_model(self, name):
        json_file = open(name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = tf.keras.models.model_from_json(loaded_model_json)


if __name__ == '__main__':
    print('test')
    my_ia = Ia('0', 'Bactest/')
    my_ia.create_model()
    print(my_ia.model.summary())
    # my_ia.compiler()
    # my_ia.fit([np.random.random(120)], [np.random.random(30)], 100)
    # a = my_ia.predict([np.random.random(120)])
    # print(a)
    # print(type(a[0]))
