import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


filename_queue = tf.train.string_input_producer(["dataHwH.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [["s"], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
col1, col2, col3, col4, col5, col6, col7, col8, col9  = tf.decode_csv(
    value, record_defaults=record_defaults)
german = tf.stack([col2, col3, col4, col5])
usa = tf.stack([col6, col7, col8, col9])
fiveYearsDEM=np.array([])
weeklyChangeDEM=np.array([])
fiveYearsUSA=np.array([])
spread=np.array([])



with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(2100):
    # Retrieve a single instance:
    date, german_curve, usa_curve = sess.run([col1, german, usa]) #, col3, col4, col5, col6, col7, col8, col9])
    fiveYearsDEM = np.append(fiveYearsDEM, german_curve[2])
    fiveYearsUSA = np.append(fiveYearsUSA, usa_curve[2])
    spread = np.append(spread, usa_curve[2]-german_curve[2])

  coord.request_stop()
  coord.join(threads)




# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.0001)
train = optimizer.minimize(loss)

# training data

x_train = fiveYearsDEM[1800:2000]
y_train = fiveYearsUSA[1800:2000]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(200):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
