import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


filename_queue = tf.train.string_input_producer(["dataHwH.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [["s"], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
col1, col2, col3, col4, col5, col6, col7, col8, col9  = tf.decode_csv(
    value, record_defaults=record_defaults)
market = tf.stack([col2, col3, col5, col6, col7, col8, col9])    #col4 is DEM 10y, the Variable to explain
#usa = tf.stack([col6, col7, col8, col9])
tenDEM = tf.stack([col4])

market_train_array=np.array([])
market_test_array=np.array([])
tenDEM_test=np.array([])
tenDEM_train=np.array([])


with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1800):
    # Retrieve a single instance:
    date_train, market_train, tenDEM_temp = sess.run([col1,market, tenDEM]) #, col3, col4, col5, col6, col7, col8, col9])
    market_train_array=np.append(market_train_array, market_train,0)
    tenDEM_train=np.append(tenDEM_train, tenDEM_temp,0)

  for i in range(1801,2100):
    date_test, market_test, tenDEM_temp = sess.run([col1, market, tenDEM])
    market_test_array=np.append(market_test_array,market_train,0 )
    tendDEM_test=np.append(tenDEM_test, tenDEM_temp,0)

  coord.request_stop()
  coord.join(threads)

print(tenDEM_train)


sess=tf.InteractiveSession()

# Model parameters
W = tf.Variable(tf.zeros([7,1]))
b = tf.Variable([0.0], dtype=tf.float32)


# Model input and output
x = tf.placeholder(tf.float32, shape=[None, 7])
y = tf.nn.softmax(tf.matmul(x, W) + b)


#training
y_ = tf.placeholder(tf.float32)
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess= tf.InteractiveSession()
tf.global_variables_initializer().run()


for i in range(200):
    x_train = market_train_array
    y_train = tenDEM_train
    sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
