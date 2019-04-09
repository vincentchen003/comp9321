import numpy as np
import re
import tensorflow as tf
data_file='data'
a = open(data_file)
List = []
tf.set_random_seed(1)
for i in a:
    i = re.sub("\n","",i)
    if re.match(".*\?.*",i) != None:
        continue
    line = i.split(",")
    line = [float(x) for x in line]
    List.append(line)
for i in range(3):
    List.append(List[i])
data = np.array(List).T
var = data[0:13].T
label = data[-1].T
var = np.insert(var,0,np.ones(300),axis = -1)
var = np.mat(var)
print(var)
print(label)
var = var.astype(np.float32)
Weight = tf.Variable(tf.random_uniform([14],0.0,0.5))
y = tf.multiply(var,Weight)
y = tf.reduce_sum(y,axis=1)
loss = tf.reduce_mean(tf.square(y-label))
optimizer = tf.train.GradientDescentOptimizer(0.0000001)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(1801):
    sess.run(train)
    if step % 100 == 0:
        print(step,sess.run(Weight))
