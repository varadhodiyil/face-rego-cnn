
import logging
import sys

import numpy as np
import tensorflow as tf

import cv2
import data_utils
from model import CNNBuilder


def load_training_data_CNN(train_path, y_val):
    """
        Load training data from path
    """

    train_data_x = []
    train_data_y = []

    _, _, data_list = data_utils.get_file_list(train_path)
    
    # s = min(100000, len(data_list))
    s = len(data_list)
    for i in range(s):
        sys.stdout.write('\r  >> Loading training data from %s (%d/%d)' % (train_path, i+1, s))
        sys.stdout.flush()

        image_data = cv2.imread(data_list[i], cv2.IMREAD_GRAYSCALE)
        if image_data is not None:
            training_data = image_data.reshape(100, 100, 1)
            train_data_x.append(training_data)
            train_data_y.append(data_utils.expand(y_val, 3))
            
    print("\n  Total count of training data : %d" % len(train_data_x))

    return train_data_x, train_data_y


if __name__ == "__main__":

    """ ------------------ Input argument process --------------------- """
    in_arg = ['1000',               # training step
              '0.001',              # learning rate
              '0.8',                # convolution parameter for CNN
              '0.5',                # hidden parameter for CNN
              'train_faces']        # input train faces

    for arg_ind in range(len(sys.argv) - 1):
        in_arg[arg_ind] = sys.argv[arg_ind + 1]

    step = int(in_arg[0])
    para_rate = float(in_arg[1])
    p_conv = float(in_arg[2])
    p_hidden = float(in_arg[3])

    

    """ ------------ Loading the training data from selected path --------------- """
    model_name = 'models/face_CNN'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(message)s',
                        datefmt='%d/%b/%Y %H:%M:%S',
                        )

    data_utils.log_print(logging, "Loading training data ...")
    [train_x1, train_y1] = load_training_data_CNN('/home/madhan/Documents/skalenow/face_reco/train_faces/madhan', 0)
    [train_x2, train_y2] = [] , []
    # [train_x1, train_y1] = load_training_data_CNN('train_faces/0', 0)
    # [train_x2, train_y2] = load_training_data_CNN('train_faces/1', 1)
    [train_x3, train_y3] = load_training_data_CNN('/home/madhan/Documents/skalenow/face_reco/train_faces/sachin', 1)
    [train_x4, train_y4] = load_training_data_CNN('/home/madhan/Documents/skalenow/face_reco/train_faces/sanath', 2)
    # [train_x2, train_y2] = load_training_data_CNN('training_face_rec', 1)

    """ --------------------- Configuration of CNN model ------------------------ """
    print("Configuration of CNN model ...")
    _cnn = CNNBuilder()
    py_x, p_keep_hidden, p_keep_con, X, Y = _cnn.model_CNN()
    cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.AdamOptimizer(para_rate).minimize(cost_op)
    predict_op = tf.argmax(py_x, 1)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    """ ------------Loading model weight from previous training result----------- """
    print("Loading model weight from previous training result ...")
    try:
        saver.restore(sess, model_name)
    except:
        pass

    """ ---------------------------- Training data ------------------------------ """
    data_utils.log_print(logging, "Training data ...")

    train_x = train_x1 + train_x2+ train_x3 + train_x4
    train_y = train_y1 + train_y2 + train_y3 + train_y4
    # train_x = train_x1 + train_x2 
    # train_y = train_y1 + train_y2 
    data_len = train_y.__len__()

    batch_size = 1000
    acc = 0

    for step_i in range(step + 1):
        predict_y = []
        cost = 0
        for sub_step in range(0, data_len, batch_size):
            tr_x = train_x[sub_step:sub_step + batch_size]
            tr_y = train_y[sub_step:sub_step + batch_size]
            
            if step_i % 10 == 0:
                ret_cost = sess.run(cost_op, feed_dict={X: tr_x, Y: tr_y,
                                                                    p_keep_con: p_conv, p_keep_hidden: p_hidden})
                cost += ret_cost
                ret_y = sess.run(predict_op, feed_dict={X: tr_x, Y: tr_y, p_keep_con: 1, p_keep_hidden: 1})
                predict_y = np.append(predict_y, ret_y)
            else:
                sess.run(train_op, feed_dict={X: tr_x, Y: tr_y, p_keep_con: p_conv, p_keep_hidden: p_hidden})

        if step_i % 10 == 0:
            acc = np.mean(np.argmax(train_y, axis=1) == predict_y)
            data_utils.log_print(logging, ('\r  %s: %d, %s: %f, %s: %.2f' %
                                           ("step", step_i, "cost", cost, "accuracy", acc * 100)))
            saver.save(sess, model_name)
        else:
            sys.stdout.write('\r  %s: %d' % ("step", step_i))
            sys.stdout.flush()

    print("Training Finished!")
