import tensorflow as tf
import numpy as np

x = tf.constant ( 35, name='x' )
y = tf.Variable ( x + 5, name='y' )

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run ( model )
    print ( session.run(y) )


# y=5*x^2−3*x+15 1000 examples and visualise the graph

# As a general rule, NumPy should be used for larger lists/arrays of numbers,
# as it is significantly more memory efficient and faster to compute on than lists
x1 = np.random.randint ( 100, size=10000 )
y1 = tf.Variable ( 5*tf.pow( x1, 2 ) - 3*x1 + 15 , name='y' )

with tf.Session() as session:
    merged = tf.summary.merge_all()
    # run tensorboard --logdir=D:\tmp\basic on windows cmd
    writer = tf.summary.FileWriter("/tmp/basic", graph=tf.get_default_graph())
    model =  tf.global_variables_initializer()
    session.run(model)
    print(session.run(y1))
    

