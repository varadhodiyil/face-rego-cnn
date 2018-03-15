import tensorflow as tf

class CNNBuilder():
    def __init__(self,num_classes=3,channels=1):
        self.num_classes = num_classes
        self.channels = channels # 1 -> GreyScale 3 -> RGB
    def init_weights(self, shape):
        """
            Initiation of weights of CNN model
        """
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def model_CNN(self):
        """
            Image classification CNN model
        """

        tf.reset_default_graph()

        X = tf.placeholder("float", [None, 100, 100, self.channels]) # 100*100 GreyScale Images
        Y = tf.placeholder("float", [None, self.num_classes]) # 5 labels

        w1 = self.init_weights([3, 3, 1, 16])  # 3x3x1 conv, 16 outputs
        w2 = self.init_weights([3, 3, 16, 32])  # 3x3x16 conv, 32 outputs
        w3 = self.init_weights([3, 3, 32, 64])  # 3x3x32 conv, 64 outputs
        w4 = self.init_weights([1600, 200])  # 64*5*5 input, 2000 outputs
        w5 = self.init_weights([200, self.num_classes])  # 2000 inputs, 1000 outputs (labels)

        p_keep_conv = tf.placeholder("float")
        p_keep_hidden = tf.placeholder("float")

        l1a = tf.nn.relu(tf.nn.conv2d(X, w1, strides=[1, 2, 2, 1], padding='SAME'))  # l1a shape=(?,224, 224, 32)
        l1 = tf.nn.max_pool(l1a, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')  # l1 shape=(?, 38, 38, 32)
        l1 = tf.nn.dropout(l1, p_keep_conv)

        l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))  # l2a shape=(?, 38, 38, 64)
        l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # l2 shape=(?, 19, 19, 64)
        l2 = tf.nn.dropout(l2, p_keep_conv)

        l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))  # l3a shape=(?, 19, 19, 128)
        l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # l3 shape=(?, 10, 10, 128)

        l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])  # reshape to (?, 12800)
        l3 = tf.nn.dropout(l3, p_keep_conv)

        l4 = tf.nn.relu(tf.matmul(l3, w4))
        l4 = tf.nn.dropout(l4, p_keep_hidden)

        py_x = tf.matmul(l4, w5)
        return py_x, p_keep_hidden, p_keep_conv, X, Y