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
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])

    def forward(self):
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        self.value = np.dot(X, W) + b
        
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """This method is separate from `forward` because it
        will be used later with `backward` as well."""
        sigmoid = 1. / (1. + np.exp(-x))
        return sigmoid

    def forward(self):
        x = self.inbound_nodes[0].value
        self.value = self._sigmoid(x)

class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    def forward(self):
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        m = len(y)
        cost = (1 / m) * np.sum(np.square(y - a))
        self.value = cost

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
def forward_pass(graph):
	for n in graph:
		n.forward()
	"""
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: The output node of the graph (no outgoing edges).
        `sorted_nodes`: a topologically sorted list of nodes.

    Returns the output node's value
    """




