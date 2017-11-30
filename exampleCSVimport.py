import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


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
i=0
for each in fiveYearsDEM:
    if i>5:
        weeklyChangeDEM =np.append(weeklyChangeDEM, 100*(fiveYearsDEM[i]-fiveYearsDEM[i-5]))
    i+=1


plt.plot(fiveYearsDEM)
plt.plot(weeklyChangeDEM)
plt.plot(fiveYearsUSA)
plt.plot(spread)
plt.ylabel("courbes Allemande et USA du " + str(date))
plt.show()
