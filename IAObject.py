import tensorflow as tf
import numpy as np


class Ia:
    def __init__(self, v, path, date):
        self.model = None
        self.v = v
        self.date = date
        self.path = path
        self.callbacks_list = []

    def create_model(self):
        val_input = tf.keras.layers.Input(shape=(60, 6, 4, 1), name='input')

        x = tf.keras.layers.Flatten()(val_input)
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

        val_output = tf.keras.layers.Dense(60, activation='linear')(x)

        self.model = tf.keras.models.Model(inputs=[val_input], outputs=[val_output])

    def compiler(self, loss='mean_squared_error', lr=0.01, decay=0.000_001, momentum=0.9, nesterov=True):
        sgd = tf.keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
        self.model.compile(optimizer=sgd, loss=loss, metrics=['mae', 'acc'])

    def init_callbacks(self, tensorboard=True, checkpoint=True):
        if tensorboard:
            self.callbacks_list.append(tf.keras.callbacks.TensorBoard(f'Backtest/logs/{self.v}-{self.date}', histogram_freq=1))
        if checkpoint:
            self.callbacks_list.append(tf.keras.callbacks.ModelCheckpoint(f'{self.path}'+"{epoch:03d}-{loss:.7f}.h5", monitor='loss', verbose=1, save_best_only=True, mode='min'))

    def fit(self, datas, labels, epochs=10, validation_split=0.1, batch_size=64, initial_epoch=None):
        return self.model.fit(datas,
                              labels,
                              batch_size=batch_size,
                              validation_split=validation_split,
                              epochs=epochs,
                              initial_epoch=0 if initial_epoch is None else initial_epoch,
                              verbose=1,
                              callbacks=self.callbacks_list if len(self.callbacks_list)>0 else None,
                              shuffle=True,
                              )

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
