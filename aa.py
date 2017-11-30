import tensorflow as tf
import numpy as np
W = tf.Variable(tf.zeros([7,1]))
print(W)

sess=tf.InteractiveSession()

print(tf.transpose(tf.zeros([7])).eval(session=sess))

print(tf.zeros([1,7]).eval(session=sess))
print(np.transpose(tf.zeros([7,1]).eval(session=sess)))


test=np.array([[1],[7]])

print(test)
