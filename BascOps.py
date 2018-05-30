#To learn and understand basic python opertions, syntax and working
import tensorflow as tf
import os


x = tf.constant([1.,2.,3.], dtype = 'float64', name='x' )
y = tf.constant([1.,2.,3.], dtype = 'float64', name='y')

# Add two tensors of the same type, x + y
addition = tf.add(x, y) 

#Subtract tensors of the same type, x — y
subtraction = tf.subtract(x, y) 

# Multiply two tensors element-wise
mult_elemwise = tf.multiply(x, y) 

#matrix multiplication of x and y
mult_matrix = tf.matmul( tf.expand_dims(x,0), tf.transpose(tf.expand_dims(y,0)))

z = tf.constant([3.], dtype = 'float64', name='z')
#Take the element-wise power of x to z
cube = tf.pow(x, z) 

#Equivalent to pow(e, x), where e is Euler’s number (2.718…)
exp = tf.exp(x) 

#Equivalent to pow(x, 0.5)
root = tf.sqrt(x) 

# Take the element-wise division of x and y
div_elemwise = tf.div(x, y) 

s = tf.constant([1,2,3], name='s')
t = tf.constant([1,5,6], name='t')
div_elemwise1 = tf.div(s, s) 
# Same as tf.div, except casts the arguments as a float
true_div = tf.truediv(s, t) 

#Same as truediv, except rounds down the final answer into an integer
floor_div = tf.floordiv(s, t) 

u = tf.constant([2], name='u')
#Takes the element-wise remainder from division
mod = tf.mod(s, u)

with tf.Session() as session:
    print("The value of [1., 2., 3.] + [1., 2., 3.] in session scope :" )
    print(session.run(addition))
    os.system("pause")
    print("The value of [1., 2., 3.] - [1., 2., 3.] in session scope :" )
    print(session.run(subtraction))
    os.system("pause")
    print("The value of [1., 2., 3.] * [1., 2., 3.] in session scope :" )
    print(session.run(mult_elemwise))
    print("The value of [1., 2., 3.]---> x * [[1.], [2.], [3.]]--->y in session scope :" )
    print(tf.expand_dims(x,0).shape)
    print(tf.transpose(tf.expand_dims(y,0)))
    print(session.run(mult_matrix))
    os.system("pause")
    print("The value of [1., 2., 3.] ^ [3.] in session scope :" )
    print(session.run(cube))
    os.system("pause")
    print("The value of [1., 2., 3.] exponential in session scope :" )
    print(session.run(exp))
    os.system("pause")
    print("\nThe value of [1., 2., 3.] square root result in session scope :  \n" )
    print(session.run(root))
    os.system("pause")
    print("The value of [1., 2., 3.] / [1., 2., 3.] in session scope :" )
    print(session.run(div_elemwise))
    os.system("pause")
    print("The value of [1, 2, 3] / [1, 2, 3] in session scope :" )
    print(session.run(div_elemwise1))
    os.system("pause")
    print("The value of [1, 2, 3] / [1, 5, 6]  with decimals in session scope :" )
    print(session.run(true_div))
    os.system("pause")
    print("The value of [1, 2, 3] / [1, 5, 6]  in Integer in session scope :" )
    print(session.run(floor_div))
    os.system("pause")
    print("The value of [1, 2, 3] / [2]  mod in session scope :" )
    print(session.run(mod))
    
