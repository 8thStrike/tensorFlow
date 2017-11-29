import tensorflow as tf

#filename_queue = tf.train.string_input_producer(["fle0.csv", "file1.csv"])

filename_queue = tf.train.string_input_producer(["dataHwH.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [["s"], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
col1, col2, col3, col4, col5, col6, col7, col8, col9  = tf.decode_csv(
    value, record_defaults=record_defaults)
german = tf.stack([col2, col3, col4, col5])
usa = tf.stack([col6, col7, col8, col9])

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(10):
    # Retrieve a single instance:
    date, german_curve, usa_curve = sess.run([col1, german, usa]) #, col3, col4, col5, col6, col7, col8, col9])
    print(date)
    print(german_curve)
    print(usa_curve)

  coord.request_stop()
  coord.join(threads)
