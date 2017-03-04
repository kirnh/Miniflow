import numpy as np

class Node(object):

	def __init__(self, inbound_nodes = []):
		# Node(s) from which this Node recieves values
		self.inbound_nodes = inbound_nodes
		# Node(s) to which this Node passes values
		self.outbound_nodes = []
		# For each inbound Node here, add this Node as an outbound
		# Node for _that_ Node
		for n in inbound_nodes:
			n.outbound_nodes.append(self)
		# A calculated value
		self.value = None

	def forward(self):
		"""
		Forward propagation.

		Compute the output value based on 'inbound_nodes' 
		and store the result in self.value.
		"""
		raise NotImplemented

class Input(Node):
    def __init__(self):
        # An Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator.
        Node.__init__(self)

    # NOTE: Input node is the only node where the value
    # may be passed as an argument to forward().
    #
    # All other node implementations should get the value
    # of the previous node from self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value
# Add node class that adds values passed as inputs	
class Add(Node):
	def __init__(self, *inputs):
		Node.__init__(self, inputs)

	def forward(self):
		self.value = 0
		for n in self.inbound_nodes:
			self.value = self.value + n.value

# Mul node class that multiplies values passed on to it as inputs
class Mul(Node):
	def __init__(self, *inputs):
		Node.__init__(self, inputs)

	def forward(self):
		self.value = 1
		for n in self.inbound_nodes:
			self.value = self.value * n.value

# A node that acts as a linear function
class Linear(Node):
	def __init__(self, inputs, weights, bias):
		Node.__init__(self, [inputs, weights, bias])

	def forward(self):
		self.value = bias
		self.value = np.sum(np.array(self.inbound_nodes[0].value)\
		 * np.array(self.inbound_nodes[1].value)) + self.inbound_nodes[2].value

def topological_sort(feed_dict):
    """
    Sort the nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Node and the value is the respective value feed to that Node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


# forward_pass() runs the network and outputs a value
def forward_pass(output_node, sorted_nodes):
	for n in sorted_nodes:
		n.forward()
	return output_node.value
	"""
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: The output node of the graph (no outgoing edges).
        `sorted_nodes`: a topologically sorted list of nodes.

    Returns the output node's value
    """

x, y, z, a = Input(), Input(), Input(), Input()
add = Add(x, y, z, a)
mul = Mul(x, y, z, a)
feed_dict = {x: 10, y: 20, z: 25, a: 45}
sorted_nodes = topological_sort(feed_dict = feed_dict)

print 'Testing Add node: ', forward_pass(add, sorted_nodes)
print 'Testing Mul node: ', forward_pass(mul, sorted_nodes)

print 'Testing Linear node:'
inputs, weights, bias = Input(), Input(), Input()
linear = Linear(inputs, weights, bias)
feed_dict = {
	inputs: [6, 14, 3],
	weights: [0.5, 0.25, 1.4],
	bias: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(linear, graph)

print output



