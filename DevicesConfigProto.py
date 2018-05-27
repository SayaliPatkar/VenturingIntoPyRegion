import tensorflow as tf

#Run this one on windows command prompt

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session()
# Runs the op.
print(sess.run(c))

# Creates a graph.
with tf.device('/cpu:0'):
    #by default visible GPU is added unless u specify the use of CPU only
    a1 = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a1')
    b1 = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b1')
c1 = tf.matmul(a1, b1)
# Creates a session with log_device_placement set to True.
sess1 = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
#Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3039 MB memory)
#-> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
print(sess1.run(c1))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory)
#
-> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
print(sess1.run(c1))
