import numpy as np

INITIAL_NODES = 1000

NODE_DTYPE = [('depth', 'i4'), ('children', '8i4'), ('parent', 'i4'), ('center', '3f4'), ('flagged', '?')]

OCT_SHIFT = np.zeros((8,3))

for n in range(8):
    OCT_SHIFT[n,0] = 2*(n&1) - 1
    OCT_SHIFT[n,1] = (n&2) -1
    OCT_SHIFT[n,2] = (n&4)/2.0 -1

class SDFOctree(object):
    def __init__(self, bounds, sdf, n_points=1, eps=0.01):
        """
        Octree generated from a signed distance function. Only boxes that 
        satisfy the signed distance function and point requirements are
        kept.

        Parameters
        ----------
            bounds : list
                List of bounds on the octree [xl, xu, yl, yu, zl, zu] where
                l denotes lower and u denotes upper.
            sdf : function
                A signed distance function, which accepts an array of points
                as input.
            n_points : float
                The desired number of points per nanometer
            eps : float
                The precision of the Octree approximation
        """

        self._bounds = bounds
        self._sdf = sdf
        self._density = n_points  # points/(1 nm)^3
        self._nodes = np.zeros(INITIAL_NODES, NODE_DTYPE)
        self._eps = eps

        self._xwidth = self._bounds[1]-self._bounds[0]
        self._ywidth = self._bounds[3]-self._bounds[2]
        self._zwidth = self._bounds[5]-self._bounds[4]

        self._nodes['center'][0] = [(self._bounds[1]+self._bounds[0])/2.0, 
                                    (self._bounds[3]+self._bounds[2])/2.0, 
                                    (self._bounds[5]+self._bounds[4])/2.0]

        self._next_node = 1
        self._resize_limit = INITIAL_NODES

        self.divide(0)  # Generate the octree

    def points(self):
        return self._nodes['center'][self._nodes['flagged']]

    def length(self, depth):
        scale = 2**depth    
        return self._xwidth/scale, self._ywidth/scale, self._zwidth/scale

    def long_length(self, depth):
        # Maximum distance traversable within a cube at depth
        return np.sqrt(np.sum(np.array([self.length(depth)])**2))

    def volume(self, depth):
        return np.prod(self.length(depth))

    def density(self, depth):
        # Note that the only point in an SDFOctree node is the node's center
        return 1.0/self.volume(depth)  # 1 point/unit volume

    def _resize(self):
        old_nodes = self._nodes
        new_size = int(self._nodes.shape[0]*1.5 + 0.5)
        self._nodes = np.zeros(new_size, NODE_DTYPE)
        self._nodes[:self._next_node] = old_nodes
        self._resize_limit = new_size

    def _add_node(self, depth, parent, center):
        if self._next_node == self._resize_limit:
            self._resize()
        self._nodes['depth'][self._next_node] = depth
        self._nodes['parent'][self._next_node] = parent
        self._nodes['center'][self._next_node] = center
        self._next_node += 1

    def divide(self, node_idx):
        node = self._nodes[node_idx]
        abs_dist = np.abs(self._sdf(node['center']))
        if (abs_dist < self._eps):
            node['flagged'] = 1
            return

        if (self.density(node['depth']) >= self._density):
            return
        
        for _i in np.arange(8):
            node['children'][_i] = self._next_node
            new_center = node['center'] + 0.5*OCT_SHIFT[_i]*self.length(node['depth']+1)
            self._add_node(node['depth']+1, node_idx, new_center)
            self.divide(node['children'][_i])
        