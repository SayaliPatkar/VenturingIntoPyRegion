import tensorflow as tf

a = tf.constant(3, name='a')
b = tf.constant(4, name='b')
c = a + b

# prints 'Tensor("add:0", shape=(), dtype=int32)'
print("The value of c without seesion and eval:" )
print(c)

sess = tf.Session()
print("\nThe value of c with seesion but not 'in session scope' :  %d\n" % (sess.run(c)))

c1 = tf.add(a,b)
print("The vaflue of c with eval but not 'in session scope'")
print("c1.eval() doesnt work outside session scope\n")

a1 = tf.constant([1, 2, 3], name='a1')
b1 = tf.constant([4, 5, 6], name='b1')
add_op = a1 + b1
add_op1 = a1 + b

a2 = tf.constant([[1, 2, 3], [4, 5, 6]], name='a2')
b2 = tf.constant([[1, 2, 3], [4, 5, 6]], name='b2')
add_op2 = a2 + b2
add_op3 = a2 + b
add_op4 = a2 + b1
b3 =tf.constant( [[100], [101]], name='b3')
add_op5 = a2 + b3
print(a2.shape)
print(b3.shape)

with tf.Session() as session:
    print("The value of c in session scope :  %d\n" % (session.run(c)))
    print("The value of c in session scope with eval :  %d\n" % (c1.eval()))
    print("The value of [1, 2, 3] + [4, 5, 6] in session scope :" )
    print(session.run(add_op))
    print("\nThe value of broadcasting, [1, 2, 3]+ 4 result in session scope :  \n" )
    print(session.run(add_op1))
    print("\nThe value of broadcasting, [[1, 2, 3], [4, 5, 6]]+ [[1, 2, 3], [4, 5, 6]] result in session scope :  \n" )
    print(session.run(add_op2))
    print("\nThe value of broadcasting, [[1, 2, 3], [4, 5, 6]]+ 4 result in session scope :  \n" )
    print(session.run(add_op3))
    print("\nThe value of broadcasting, [[1, 2, 3], [4, 5, 6]]+ [4, 5, 6] result in session scope :  \n" )
    print(session.run(add_op4))
    print("\n**************************************************************************\nFor broadcasting to work, both operands have some dimension in common, for eg no of rows in a = no of rows in b\n or no of columns in a = no of rows in b")
    print("It will simply not work for cross match, eg no of rows in a = no of columns in b" )
    print("[[1, 2, 3], [4, 5, 6]]+ [100, 101] will NOT work but, \n[[1, 2, 3], [4, 5, 6]]+ [[100], [101]] will give result : " )
    print(session.run(add_op5))
    print("\n**************************************************************************")



