import tensorflow as tf
hello = tf.constant("hello world")
print(hello)
# you will see a string of tensor here
print(hello.numpy())
print(hello.numpy().decode('utf-8'))

a = tf.constant(2021)
b = tf.constant(303)
c = a + b
print(c)
print(c.numpy())
print("TensorFlow version:", tf.__version__)
