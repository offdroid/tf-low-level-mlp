import tensorflow as tf


class Model:
    def __init__(self,
                 ds,
                 learning_rate,
                 training_steps,
                 tensorboard_callback=None):
        self.train_ds = ds['train']
        self.test_ds = ds['test']
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.tensorboard_callback = tensorboard_callback

        # Define the model
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(784, )),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])

        sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        # Specify optimizer and loss function
        self.model.compile(optimizer=sgd,
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])

    def train(self, epochs):
        callbacks = [self.tensorboard_callback
                     ] if self.tensorboard_callback is not None else []
        self.model.fit(self.train_ds,
                       epochs=epochs,
                       steps_per_epoch=self.training_steps,
                       callbacks=callbacks)

    def predict(self, image):
        return tf.argmax(self.model.predict(image), axis=1).numpy()[0]
