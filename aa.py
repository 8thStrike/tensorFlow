import tensorflow as tf
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
print(test)
test=np.append(test, [[1,2,3,4,5]],1) #Obligatory : axe 1 for the first vector
print(test)

test=np.append(test, [[1,2,3,4,5]],0)
print(test)
test=np.append(test, [[1,2,3,4,5]],0)
print(test)
test=np.append(test, [[1,2,3,4,5]],0)
print(test)
test=np.append(test, [[1,2,3,4,5]],0)
print(test)

#array method

print(test.lenght)
'''
test=np.append(test, [[1,2,3,4,5]],1)
print(test)
test=np.append(test, [[1,2,3,4,5]],1)
print(test)
test=np.append(test, [[1,2,3,4,5]],1)
print(test)
test=np.append(test, [[1,2,3,4,5]],1)
print(test)

#other trial
print("Other Trial with append on axis 0")
test=np.array([[]])
print(test)
test=np.append(test, [[1,2,3,4,5]])
print(test)
test=np.append(test, [[1,2,3,4,5]])
print(test)
test=np.append(test, [[1,2,3,4,5]])
print(test)
test=np.append(test, [[1,2,3,4,5]])
print(test)
test=np.append(test, [[1,2,3,4,5]])
print(test)
'''
