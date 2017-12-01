import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator

import numpy as np
'''
W = tf.Variable(tf.zeros([7,1]))
print(W)

sess=tf.InteractiveSession()

print(tf.transpose(tf.zeros([7])).eval(session=sess))

print(tf.zeros([1,7]).eval(session=sess))

print(np.transpose(tf.zeros([7,1]).eval(session=sess)))


#This is how you add vector to a vector of vector (or line to a matrix)
test=np.array([[1,45,4,34,23],[5,4,56,111,7]])

print(test)

test=np.append(test,[[1,2,3,56,987]], 0)

print(test)


'''



#Other trial
print("Other Trial with append on axis 1")
test=np.array([[]], dtype=float)
print("test.size")
print(test.size)
test=np.append(test, [[1,2,3,4,5]],1) #Obligatory : axe 1 for the first vector
test=np.append(test, [[2,2,3,4,5]],0)
test=np.append(test, [[3,2,3,4,5]],0)
test=np.append(test, [[4,2,3,4,5]],0)
test=np.append(test, [[5,2,3,4,5]],0)
test=np.append(test, [[6,2,3,4,5]],0)


val=np.array([[]], dtype=float)
val=np.append(val, [[1,5,5,6,7]],1)
val=np.append(val, [[2,5,5,6,7]],0)
val=np.append(val, [[3,5,5,6,7]],0)
val=np.append(val, [[4,5,5,6,7]],0)
val=np.append(val, [[5,5,5,6,7]],0)
val=np.append(val, [[6,5,5,6,7]],0)


print(test )
print(val)

#*****************************************************************************

'''
let's try to build a dataset from tf.dataset => and then use it in some simple loop
'''

train_x = tf.constant(test)
train_labels = tf.constant([11,12,13,14,15,16])

val_x = tf.constant(val)
val_labes = tf.constant([21,22,23,24,25,26])

#create tensorflow dataset objects
train_data = tf.data.Dataset.from_tensor_slices((train_x, train_labels))
val_data = tf.data.Dataset.from_tensor_slices((val_x, val_labes))


#create tensorflow Iterator object
iterator=Iterator.from_structure(train_data.output_types, train_data.output_shapes)
next_element=iterator.get_next()


#create two initialization ops to switch between the datasets.
training_init_op=iterator.make_initializer(train_data)
validation_init_op=iterator.make_initializer(val_data)


with tf.Session() as sess:

    #initialize the iterator
    sess.run(training_init_op)

    #get each element of the training dataset until the end is reached
    while True:
        try:
            elem=sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("end of training dataset")
            break

    #initialize the iterator on  the validation data
    sess.run(validation_init_op)

    #get each element of the validation dataset until the end is reached
    while True:
        try:
            elem=sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("end of validation dataset.")
            break
