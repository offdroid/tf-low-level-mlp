import tensorflow as tf


class Model:
    def __init__(self, ds, learning_rate, training_steps):
        self.train_iter = ds['train_iter']
        self.test_iter = ds['test_iter']
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        # Weights for hidden and output layer
        self.W = [
            tf.Variable(tf.random.truncated_normal([784, 128], stddev=0.1)),
            tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
        ]
        # Biases
        self.b = [
            tf.Variable(tf.random.truncated_normal([128], stddev=0.1)),
            tf.Variable(tf.random.truncated_normal([10], stddev=0.1))
        ]
        # Optimizer
        self.sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    # Note the use of `tf.function`. This instructs TF to construct a graph from
    # the function, which can help with performance. However, such annotated functions
    # can behave different than regular Python functions if not used properly
    @tf.function
    def __step(self, images, labels, train=True):
        '''Perform a (training) step on the provided data
         returning the loss and probability'''
        with tf.GradientTape() as tape:
            # Compute logits for the hidden and output layer
            l1 = tf.nn.relu(tf.matmul(images, self.W[0]) + self.b[0])
            l2 = tf.matmul(l1, self.W[1]) + self.b[1]

            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l2,
                                                               labels=labels))
            probability = tf.nn.softmax(l2)

        if train:
            # Compute and apply the gradient
            # Alternatively, naive manual optmizer with
            # `W[0].assign_sub(learning_rate * grads[0])`
            grads = tape.gradient(loss, self.W + self.b)
            self.sgd.apply_gradients(zip(grads, self.W + self.b))

        return loss, probability

    @tf.function
    def predict(self, image):
        '''Predict/Classify a single provided image'''
        # Similar to `step()` but without providing labels
        l1 = tf.nn.relu(tf.matmul(image, self.W[0]) + self.b[0])
        l2 = tf.matmul(l1, self.W[1]) + self.b[1]
        probability = tf.nn.softmax(l2)
        # The label with the highest probability
        # our best guess
        return tf.argmax(probability, axis=1)[0]

    @tf.function
    def __measure_accuracy(self):
        '''Measure the models accuracy on a batch from
         the test dataset'''
        test_images, test_label = next(self.test_iter)
        # Perform a step on the test data without
        # training the model
        loss, probability = self.__step(test_images, test_label, train=False)
        # Find the most likely label for each image
        predictions = tf.argmax(probability, axis=1, output_type=tf.int32)
        # Compare the guessed with the actual labels
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions, test_label), tf.float32))
        return loss, accuracy

    def __log(self, epoch, epochs, loss, accuracy):
        '''Print out information to the console and summary writer'''
        # Print out to the console
        print(
            f"Epoch {epoch}/{epochs}: loss={loss:.4f}" \
            f" - accuracy={accuracy:.4f}"
        )
        # Log to the summary writer for the data to be
        # accessible through TensorBoard
        tf.summary.scalar('epoch_accuracy', data=accuracy, step=epoch)
        tf.summary.scalar('epoch_loss', data=loss, step=epoch)

    def train(self, epochs):
        '''Train the model for the number of specified epochs and log the accuracy of each epoch'''
        for i in range(self.training_steps * epochs):
            # Fetch a new batch of examples and train on them
            image_batch, label_batch = next(self.train_iter)
            self.__step(image_batch, label_batch, train=True)

            if i % self.training_steps == 0 and i / self.training_steps >= 1:
                # Measure performance (accuracy) each epoch
                loss, accuracy = self.__measure_accuracy()
                self.__log(i // self.training_steps, epochs, loss, accuracy)

        loss, accuracy = self.__measure_accuracy()
        self.__log(epochs, epochs, loss, accuracy)
