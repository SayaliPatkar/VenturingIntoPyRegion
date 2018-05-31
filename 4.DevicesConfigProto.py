import tensorflow as tf
import os
#Run this one on windows command prompt

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

config1 = tf.ConfigProto()
config1.gpu_options.per_process_gpu_memory_fraction = 0.4
with tf.Session(config = config1) as sess:
    print("**********************************************When device log placement is false**********************************")
    print("**********************************************When you specifically choose amount of memory to be allocated to process**********************************")

    # Runs the op.s
    print(sess.run(c))
os.system("pause")
# Creates a graph.

# Creates a session with log_device_placement set to True.
# enabling GPU/CPU usage log for every commands
with tf.device('/cpu:0'):
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess1:
        print("**********************************************When you specifically choose CPU over visible GPUs**********************************")
        print("**********************************************When device log placement is true**********************************")
        #by default visible GPU is added unless u specify the use of CPU only
        a1 = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a1')
        b1 = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b1')
        c1 = tf.matmul(a1, b1)
        #Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3039 MB memory)
        #-> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
        print(sess1.run(c1))
        os.system("pause")

config = tf.ConfigProto()
#allow minimum possible memory from the start
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    print("**********************************************When GPU capacity is allocated as it is needed**********************************")
    a1 = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a1')
    b1 = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b1')
    c1 = tf.matmul(a1, b1)
    #Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory)
    #-> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
    print(session.run(c1))
