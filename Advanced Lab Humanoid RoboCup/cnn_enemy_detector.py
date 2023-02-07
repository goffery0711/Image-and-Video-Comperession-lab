import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt


class CNN_model:

    def __init__(self, input_shape, cnn_filters, drop_out, fc_filters, opt, loss, metrics):

        self.model = tf.keras.models.Sequential()
        self.input_shape = input_shape
        self.drop_out = drop_out
        self.cnn_filters = cnn_filters
        self.fc_filters = fc_filters
        self.optimizer = None
        self.opt = opt
        self.loss = loss
        self.metrics = metrics

    def build_model(self):

        # loop over all filters
        for idx, f in enumerate(self.cnn_filters):
            # add first conv layer with input shape
            if idx == 0:
                self.model.add(tf.keras.layers.Conv2D(f, (3, 3), input_shape=self.input_shape))
            else:
                self.model.add(tf.keras.layers.Conv2D(f, (3, 3)))

            self.model.add(tf.keras.layers.Activation('relu'))
            # tensorflow has trouble freezing this layer - solved
            self.model.add(tf.keras.layers.BatchNormalization(axis=-1))
            self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
            self.model.add(tf.keras.layers.Dropout(rate=self.drop_out))

        # flatten
        # opencv has trouble interpreting this layer
        #self.model.add(tf.keras.layers.Flatten())
        # global avg pool - works better anyway
        self.model.add(tf.keras.layers.GlobalAveragePooling2D())
        # Dropout
        self.model.add(tf.keras.layers.Dropout(rate=self.drop_out))

        # create fully connected NN
        for idx, f in enumerate(self.fc_filters):
            self.model.add(tf.keras.layers.Dense(f))
            if idx < len(self.fc_filters)-1:
                # if not last filter use relu and add drop out
                self.model.add(tf.keras.layers.Activation('relu'))
                self.model.add(tf.keras.layers.Dropout(rate=self.drop_out))
            else:
                # if last layer
                self.model.add(tf.keras.layers.Activation('sigmoid'))

        return self.model

    def model_compile(self):

        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=self.metrics)

    def model_summary(self):
        return self.model.summary()

    def get_optimizer(self, lr, decay):
        if self.opt == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(lr=lr, decay=decay)
        elif self.opt == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(lr=lr)

    def draw_result(self, _history):

        # Plot the loss and accuracy curves for training and validation
        plt.subplot(2, 1, 1)
        plt.plot(_history.history['loss'], color='b', label="Training loss")
        plt.plot(_history.history['val_loss'], color='r', label="Validation loss")
        plt.title('binary_crossentropy')
        plt.legend(loc='best', shadow=True)

        plt.subplot(2, 1, 2)
        plt.plot(_history.history[self.metrics[0]], color='b', label="Training metric")
        plt.plot(_history.history['val_' + self.metrics[0]], color='r', label="Validation metric")
        plt.title('accuracy')
        plt.legend(loc='best', shadow=True)

        plt.show()

    def save_model(self, cnn_file, num, save_as_h5=True, save_json=False):
        # save model as h5
        if save_as_h5:
            self.model.save(cnn_file + str(num) + '.h5')
        if save_json:
            # save model network as json
            model_json = self.model.to_json()
            with open(cnn_file + str(num) + '.json', "w") as json_file:
                json_file.write(model_json)

            self.model.save_weights(cnn_file + str(num) + 'weights.h5')

    @staticmethod
    def load_model(cnn_file, num):
        return tf.keras.models.load_model(cnn_file + str(num) + '.h5')


