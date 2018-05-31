import tensorflow as tf
import numpy as np
import os

#---------------------------------------------Constants---------------------------------------------

v = tf.constant (35, name='v' )
# Array of shape [2,3], having same variable value 4
w = tf.constant(4, dtype = 'float64', name='w',shape= [2,3] )
x = tf.constant([1.,2.,3.], shape=[3,1], dtype = 'float64', name='x' )
# if you dont give shape explicitly [4., 5. ,6.] it will be initialized with shape=(3,)
# this is proper way to do it
y = tf.constant([[4.,5.,6.]], dtype = 'float64', name='y')

#---------------------------------------------Variables---------------------------------------------
t = tf.Variable ( v + 5, name='t' )
# empty
emp = tf.Variable([3,3], dtype = tf.float32, name='emp')
#very handy for weight initialization, as 0 initialization of weight is never desorable
weights = tf.Variable(tf.random_normal([3,3], mean=0.0, stddev=0.35, dtype=tf.float64),
                      name="weights" )
zero_var1 = tf.zeros([2,2], dtype= tf.float64, name = "zero_var1")
zero_var2 = tf.zeros_like(weights, name = "zero_var2")

# y=5*x^2−3*x+15 1000 examples and visualise the graph
# As a general rule, NumPy should be used for larger lists/arrays of numbers,
# as it is significantly more memory efficient and faster to compute on than lists
x1 = np.random.randint ( 100, size=10000 )
# write a code to operation first and actually calculate later
y1 = tf.Variable ( 5*tf.pow( x1, 2 ) - 3*x1 + 15 , name='y' )

init_op = tf.global_variables_initializer()

#---------------------------------------------Placeholder-----------------------------------------------
#for placeholder value needs to be feeded as we run session
placeholder_var = tf.placeholder(tf.float32, shape=(2, 2))
mul = tf.matmul(placeholder_var, placeholder_var)


with tf.Session() as session:
    print("The value of w is : ", session.run(w))
    print("The value of x is : ", session.run(x))
    print("The value of x is : ", x)

    print("The value of y is : ", session.run(y))
    print("The value of y is : ", y)

    print("The value of v is : ", session.run(v))
    print("The value of v is : ", v)

    # All the variables are uninitialized here , hence following print will not work
    #print("The value of weights is : ", session.run(weights))
    #print("The value of emp is : ", session.run(emp))

    session.run(init_op)
    print("The value of zero_var1 is : ", zero_var1)
    print("The value of zero_var1 is : ", session.run(zero_var1))
    print("The value of zero_var2 is : ", session.run(zero_var2))
    print("The value of weights is : ", session.run(weights))
    print("The value of emp is : ", session.run(emp))


    os.system("pause")
    n = np.array([[1,2],[3,4]])
    result = session.run([mul],feed_dict={ placeholder_var : n})
    print("The value of [[1,2],[3,4]] * [[1,2],[3,4]] in session scope : ", result )

    os.system("pause")
    merged = tf.summary.merge_all()
    # run tensorboard --logdir=D:\tmp\basic on windows cmd
    writer = tf.summary.FileWriter("/tmp/basic", graph=tf.get_default_graph())

    print(session.run(y1))
