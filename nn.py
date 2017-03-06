from miniflow import *

##################################################################
#Testing working of Add and Mul node classes
x, y, z, a = Input(), Input(), Input(), Input()
add = Add(x, y, z, a)
mul = Mul(x, y, z, a)
feed_dict = {x: 10, y: 20, z: 25, a: 45}
graph = topological_sort(feed_dict)
forward_pass(graph)

print('Testing Add node output: ', add.value)
print('Testing Mul node output: ', mul.value)

###################################################################
#Testing the working of Linear Node class
print('Testing Linear node output:')
inputs, weights, bias = Input(), Input(), Input()
linear = Linear(inputs, weights, bias)
feed_dict = {
	inputs: [6, 14, 3],
	weights: [0.5, 0.25, 1.4],
	bias: 2
}

graph = topological_sort(feed_dict)
forward_and_backward(graph)
print(linear.value)

X, W, b = Input(), Input(), Input()

f = Linear(X, W, b)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

feed_dict = {X: X_, W: W_, b: b_}
graph = topological_sort(feed_dict)
forward_pass(graph)
print(f.value)

###################################################################
#Testing the working of Sigmoid activation function
print('Testing Sigmoid activation:')
X, W, b = Input(), Input(), Input()

f = Linear(X, W, b)
g = Sigmoid(f)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

feed_dict = {X: X_, W: W_, b: b_}

graph = topological_sort(feed_dict)
forward_and_backward(graph)
print(g.value)

"""
Output should be:
[[  1.23394576e-04   9.82013790e-01]
 [  1.23394576e-04   9.82013790e-01]]
"""

print("Checking Mean squared error implementation:")
y, a = Input(), Input()
cost = MSE(y, a)

y_ = np.array([1, 2, 3])
a_ = np.array([4.5, 5, 10])

feed_dict = {y: y_, a: a_}
graph = topological_sort(feed_dict)

forward_and_backward(graph)
print(cost.value)