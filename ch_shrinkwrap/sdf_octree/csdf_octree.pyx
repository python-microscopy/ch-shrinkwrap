cimport numpy as np
import numpy as np
cimport cython

INITIAL_NODES = 1000

NODE_DTYPE = [('depth', 'i4'), ('children', '8i4'), ('parent', 'i4'), ('center', '3f4'), ('flagged', 'i1')]

NODE_DTYPE2 = [('depth', 'i4'), 
               ('child0', 'i4'),
               ('child1', 'i4'),
               ('child2', 'i4'),
               ('child3', 'i4'),
               ('child4', 'i4'),
               ('child5', 'i4'),
               ('child6', 'i4'),
               ('child7', 'i4'), 
               ('parent', 'i4'), 
               ('center_x', 'f4'),
               ('center_y', 'f4'),
               ('center_z', 'f4'), 
               ('flagged', 'i1')]  # switched to int8 over bool as cython numpy lacks arrays of booleans


# OCT_SHIFT cribbed from PYME.experimental._octree
cdef float[8] _oct_shift_x
cdef float[8] _oct_shift_y
cdef float[8] _oct_shift_z
cdef int n

for n in range(8):
    _oct_shift_x[n] = 0.5*(2*(n&1) - 1)
    _oct_shift_y[n] = 0.5*((n&2) -1)
    _oct_shift_z[n] = 0.5*((n&4)/2.0 -1)

cdef packed struct node_d:
    np.int32_t depth
    np.int32_t child0
    np.int32_t child1
    np.int32_t child2
    np.int32_t child3
    np.int32_t child4
    np.int32_t child5
    np.int32_t child6
    np.int32_t child7
    np.int32_t parent
    np.float32_t center_x
    np.float32_t center_y
    np.float32_t center_z
    np.int8_t flagged

cdef class cSDFOctree(object):
    cdef np.float32_t[6] _bounds
    cdef object _sdf
    cdef public object _nodes
    cdef node_d *_cnodes
    cdef object _flat_nodes
    cdef np.float32_t _eps, _xwidth, _ywidth, _zwidth, _density
    cdef int _next_node, _resize_limit

    cdef object _sdf_arr

    cdef np.float32_t[20] _long_lengths
    cdef np.float32_t[20] _volumes
    cdef np.float32_t[20] _densities
    
    def __init__(self, bounds, sdf, n_points=1, eps=0.0001):
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
        cdef int _i

        bounds = np.array(bounds, 'f4')
        self._bounds = bounds
        self._sdf = sdf
        self._density = n_points  # points/unit volume
        self._nodes = np.zeros(INITIAL_NODES, NODE_DTYPE)
        self._eps = eps

        self._xwidth = self._bounds[1]-self._bounds[0]
        self._ywidth = self._bounds[3]-self._bounds[2]
        self._zwidth = self._bounds[5]-self._bounds[4]

        self._nodes['center'][0] = [(self._bounds[1]+self._bounds[0])/2.0, 
                                    (self._bounds[3]+self._bounds[2])/2.0, 
                                    (self._bounds[5]+self._bounds[4])/2.0]
        self._flat_nodes = self._nodes.view(NODE_DTYPE2)
        self._set_cnodes(self._flat_nodes)
        self._sdf_arr = np.zeros(3,np.float32)

        self._next_node = 1
        self._resize_limit = INITIAL_NODES

        # Precalculate the long lengths, volumes, and densities
        # for use later
        for _i in np.arange(20):
            l = self.length(_i)
            self._long_lengths[_i] = np.sqrt(np.sum(np.array([l])**2))
            self._volumes[_i] = np.prod(l)
            # This creates problems at great depths. Maybe switch to volume only?
            # That is, density is just 1/volume, so if we switch our initial density
            # to 1/n_points and compare volumes to that, we should be good.
            self._densities[_i] = 1.0/self._volumes[_i]

        self.divide()  # Generate the octree

    def points(self):
        return self._nodes['center'][self._nodes['flagged']==1]

    def length(self, np.int32_t depth):
        cdef np.int32_t scale
        scale = 2**depth
        return self._xwidth/scale, self._ywidth/scale, self._zwidth/scale

    def long_length(self, np.int32_t depth):
        if depth < 20:
            return self._long_lengths[depth]
        # Maximum distance traversable within a cube at depth
        return np.sqrt(np.sum(np.array([self.length(depth)])**2))

    def volume(self, np.int32_t depth):
        if depth < 20:
            return self._volumes[depth]
        return np.prod(self.length(depth))

    def density(self, np.int32_t depth):
        if depth < 20:
            return self._densities[depth]
        # Note that the only point in an SDFOctree node is the node's center
        return 1.0/self.volume(depth)  # 1 point/unit volume

    def _set_cnodes(self, node_d[:] nodes):
        self._cnodes = &nodes[0]

    def _resize(self):
        cdef int new_size
        old_nodes = self._nodes
        new_size = int(self._nodes.shape[0]*1.5 + 0.5)
        self._nodes = np.zeros(new_size, NODE_DTYPE)
        self._flat_nodes = self._nodes.view(NODE_DTYPE2)
        self._set_cnodes(self._flat_nodes)
        self._nodes[:self._next_node] = old_nodes[:self._next_node]
        self._resize_limit = new_size

    def _add_node(self, np.int32_t depth, np.int32_t parent, np.float32_t center_x, np.float32_t center_y, np.float32_t center_z):
        cdef node_d *node
        if self._next_node == self._resize_limit:
            self._resize()
        node = &self._cnodes[self._next_node]
        node.depth = depth
        node.parent = parent
        node.center_x = center_x
        node.center_y = center_y
        node.center_z = center_z
        self._next_node += 1

    def divide(self):
        cdef int _i, node_idx
        cdef np.float32_t dist, ll2, dpos, dneg, dpdn, new_center_x, new_center_y, new_center_z, lx, ly, lz
        cdef node_d *node
        cdef bint density_check
        
        node_idx = 0
        while node_idx < self._next_node:
            node = &self._cnodes[node_idx]

            if (node_idx > 0) and (node.depth == 0):
                # We've somehow hit the empty node zone (we shouldn't be able to do this)
                raise RuntimeError('Made it to the other world.')
            
            # Distances we need
            self._sdf_arr[0] = node.center_x
            self._sdf_arr[1] = node.center_y
            self._sdf_arr[2] = node.center_z
            dist = self._sdf(self._sdf_arr)
            ll2 = 0.5*self.long_length(node.depth)
            dpos = dist+ll2
            dneg = dist-ll2
            dpdn = dpos*dneg

            node_idx += 1

            # Voxel density check
            density_check = (self.density(node.depth) >= self._density)
            if (np.abs(dist) <= self._eps) or (density_check and (dpdn < 0)):
                # This voxel's center is acceptably close to the object defined
                # by self._sdf
                node.flagged = 1
                continue

            if density_check:
                # We've subdivided finely enough, but this voxel isn't near the
                # object defined by self._sdf
                continue

            if (dpdn > 0) and (np.abs(dpos)>self._eps) and (np.abs(dneg)>self._eps):
                # This box will never straddle the boundary of the sdf 
                continue

            for _i in range(8):
                # subdivide
                lx, ly, lz = self.length(node.depth+1)
                new_center_x = node.center_x + _oct_shift_x[_i]*lx
                new_center_y = node.center_y + _oct_shift_y[_i]*ly
                new_center_z = node.center_z + _oct_shift_z[_i]*lz
                children = &node.child0
                children[_i] = self._next_node
                self._add_node(node.depth+1, node_idx-1, new_center_x, new_center_y, new_center_z)
            