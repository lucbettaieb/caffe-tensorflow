#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from mynet import MyNet

data_node = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
net = MyNet({'data': data_node})
with tf.Session() as sess:
    output_graph = sess._graph
    net.load(data_path='mynet.npy', session=sess)
    f = tf.gfile.FastGFile('mynet.pb', "w")
    f.write(output_graph.as_graph_def().SerializeToString())
    f.close()
