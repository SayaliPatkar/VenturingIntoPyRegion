#To learn and understand basic python opertions, syntax and working
import tensorflow as tf
import os
import numpy as np

#---------------------------------------------Constants---------------------------------------------
w = tf.constant(4, dtype = 'float64', name='w',shape= [2,3] )
x = tf.constant([1.,2.,3.], shape=[3,1], dtype = 'float64', name='x' )
# if you dont give shape explicitly [4., 5. ,6.] it will be initialized with shape=(3,)
# this is proper way to do it
y = tf.constant([[4.,5.,6.]], dtype = 'float64', name='y')

#---------------------------------------------Variables---------------------------------------------
# empty
emp = tf.Variable([3,3], dtype = tf.float32, name='emp')
#very handy for weight initialization, as 0 initialization of weight is never desorable
weights = tf.Variable(tf.random_normal([3,3], mean=0.0, stddev=0.35, dtype=tf.float64),
                      name="weights" )
zero_var1 = tf.zeros([2,2], dtype= tf.float64, name = "zero_var1")
zero_var2 = tf.zeros_like(weights, name = "zero_var2")

init_op = tf.global_variables_initializer()

#---------------------------------------------Placeholder-----------------------------------------------
#for placeholder value needs to be feeded as we run session
placeholder_var = tf.placeholder(tf.float32, shape=(2, 2))
mul = tf.matmul(placeholder_var, placeholder_var)

#---------------------------------------------Addition-------------------------------------------------
# Add two tensors of the same type, x + y
addition = tf.add(x, x)
addition_broadcasting = tf.add(x, y)

#---------------------------------------------Subtraction-------------------------------------------------
#Subtract tensors of the same type, x — y
subtraction = tf.subtract(y, y)

#---------------------------------------------Multiplication-------------------------------------------------
# Multiply two tensors element-wise
mult_elemwise = tf.multiply(y, y)
#matrix multiplication of x and y
matmul_result = tf.matmul(x , tf.transpose(x))
matmul_result1 = tf.matmul(x , y)

#---------------------------------------------Division-------------------------------------------------
# Take the element-wise division of x and y
div_elemwise = tf.div(x, x)
# Same as tf.div, except casts the arguments as a float
s = tf.constant([1,2,3], name='s')
t = tf.constant([1,5,6], name='t')
true_div = tf.truediv(s, t)
#Same as truediv, except rounds down the final answer into an integer
floor_div = tf.floordiv(s, t)
u = tf.constant([2], name='u')
#Takes the element-wise remainder from division
mod = tf.mod(s, u)

#---------------------------------------------Power Ops-------------------------------------------------
z = tf.constant([3.], dtype = 'float64', name='z')
#Take the element-wise power of x to z
cube = tf.pow(x, z)
#Equivalent to pow(e, x), where e is Euler’s number (2.718…)
exp = tf.exp(x)
#Equivalent to pow(x, 0.5)
root = tf.sqrt(x)

with tf.Session() as session:
    print("The value of w is : ", session.run(w))
    print("The value of x is : ", session.run(x))
    print("The value of x is : ", x)

    print("The value of y is : ", session.run(y))
    print("The value of y is : ", y)

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
    print("The value of [[1.], [2.], [3.]] + [[1.], [2.], [3.]] in session scope : ", session.run(addition))
    print("Example of Boradcasting")
    print("The value of [[1.], [2.], [3.]] + [[4. ,5. ,6.]] in session scope : ", session.run(addition_broadcasting))
    os.system("pause")
    print("The value of [[4.,5.,6.]]- [[4.,5.,6.]] in session scope : " , session.run(subtraction))
    os.system("pause")
    print("The value of [[4.,5.,6.]] * [[4.,5.,6.]] in session scope : " , session.run(mult_elemwise))
    print("\n\nThe value of [[1.], [2.], [3.]]---> x * [[1., 2., 3.]]--->x' in session scope :", session.run(matmul_result))
    print("\n\nThe value of [[1.], [2.], [3.]]---> x * [[4., 5., 6.]]--->y in session scope :", session.run(matmul_result1))
    os.system("pause")
    print("The value of [[1.], [2.], [3.]] / [[1.], [2.], [3.]] in session scope : " , session.run(div_elemwise))
    print("\n\nThe value of [1, 2, 3] / [1, 5, 6]  with decimals in session scope : ", session.run(true_div))
    print("\n\nThe floored value of [1, 2, 3] / [1, 5, 6]  in Integer in session scope : " ,session.run(floor_div))
    os.system("pause")
    print("The value of [1., 2., 3.] ^ [3.] in session scope : " , session.run(cube))
    os.system("pause")
    print("The value of [1., 2., 3.] exponential in session scope : " , session.run(exp))
    os.system("pause")
    print("\nThe value of [1., 2., 3.] square root result in session scope : ", session.run(root))
    os.system("pause")
    print("The value of [1, 2, 3] / [2]  mod in session scope :" )
    print(session.run(mod))
