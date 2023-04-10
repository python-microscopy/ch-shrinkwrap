cimport numpy as np
import numpy as np
import scipy.spatial
import cython
import math

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from ._membrane_mesh cimport MembraneMesh
from ._membrane_mesh import MembraneMesh

from ._membrane_mesh cimport points_t
from PYME.experimental._triangle_mesh cimport NEIGHBORSIZE, VECTORSIZE, face_d, face_t, vertex_d, vertex_t, halfedge_t

#cdef const float PI 3.1415927

DEF PI=3.1415927

cdef extern from "triangle_mesh_utils.c":
    void _update_face_normals(np.int32_t *f_idxs, halfedge_t *halfedges, vertex_t *vertices, face_t *faces, signed int n_idxs)
    void update_face_normal(int f_idx, halfedge_t *halfedges, vertex_d *vertices, face_d *faces)
    void update_single_vertex_neighbours(int v_idx, halfedge_t *halfedges, vertex_d *vertices, face_d *faces)

cdef class SkeletonMesh(MembraneMesh):
    def __init__(self, vertices=None, faces=None, mesh=None, **kwargs):
        MembraneMesh.__init__(self, vertices, faces, mesh, **kwargs)


    cdef int skeleton_edge_split(self, np.int32_t _curr, np.int32_t * new_edges, np.int32_t * new_vertices, np.int32_t * new_faces, int n_edge_idx, int n_vertex_idx, int n_face_idx,  
                            bint live_update=1, bint upsample=0):
        """
        Split triangles evenly along an edge specified by halfedge index _curr.

        Parameters
        ----------
            _curr : int
                Pointer to halfedge defining edge to split.
            live_update : bool
                Update associated faces and vertices after split. Set to False
                to handle this externally (useful if operating on multiple, 
                disjoint edges).
            upsample: bool
                Are we doing loop subdivision? If so, keep track of all edges
                incident on both a new vertex and an old verex that do not
                split an existing edge.
        """
        cdef halfedge_t *curr_edge
        cdef halfedge_t *twin_edge
        cdef np.int32_t _prev, _twin, _next, _twin_prev, _twin_next, _face_1_idx, _face_2_idx, _he_0_idx, _he_1_idx, _he_2_idx, _he_3_idx, _he_4_idx, _he_5_idx, _vertex_idx
        cdef bint interior
        cdef np.int32_t v0, v1
        cdef int i
        cdef np.float32_t x0x, x0y, x0z, x1x, x1y, x1z, n0x, n0y, n0z, n1x, n1y, n1z, ndot
        cdef np.float32_t[VECTORSIZE] _vertex

        if _curr == -1:
            return 0

        if self._chalfedges[_curr].locally_manifold == 0:
            return 0
        
        curr_edge = &self._chalfedges[_curr]
        _prev = curr_edge.prev
        _next = curr_edge.next

        # Grab the new vertex position
        v0 = curr_edge.vertex
        v1 = self._chalfedges[_prev].vertex
        x0x = self._cvertices[v0].position0
        x0y = self._cvertices[v0].position1
        x0z = self._cvertices[v0].position2
        x1x = self._cvertices[v1].position0
        x1y = self._cvertices[v1].position1
        x1z = self._cvertices[v1].position2

        _vertex_idx = new_vertices[n_vertex_idx]

        # FROM HERE

        # displacement along the x1-x0 direction
        #print(curr_edge.length)
        if curr_edge.length > 1e-6:
            # project the _next vertex perpendicularly onto the edge
            n0x = x1x - x0x
            n0y = x1y - x0y
            n0z = x1z - x0z
            n1x = self._cvertices[self._chalfedges[_next].vertex].position0 - x0x
            n1y = self._cvertices[self._chalfedges[_next].vertex].position1 - x0y
            n1z = self._cvertices[self._chalfedges[_next].vertex].position2 - x0z
            ndot = ((n0x*n1x)+(n0y*n1y)+(n0z*n1z))/(curr_edge.length*curr_edge.length)
        else:
            ndot = 0.5

        _vertex[0] = x0x + (x1x-x0x)*ndot
        _vertex[1] = x0y + (x1y-x0y)*ndot
        _vertex[2] = x0z + (x1z-x0z)*ndot

        # TO HERE

        self._cvertices[_vertex_idx].position0 = _vertex[0]
        self._cvertices[_vertex_idx].position1 = _vertex[1]
        self._cvertices[_vertex_idx].position2 = _vertex[2]
        #_vertex_idx = self._new_vertex(_vertex)

        _twin = curr_edge.twin
        interior = (_twin != -1)  # Are we on a boundary?
        
        if interior:
            twin_edge = &self._chalfedges[_twin]
            _twin_prev = twin_edge.prev
            _twin_next = twin_edge.next
        
        # Ensure the original faces have the correct pointers and add two new faces
        self._cfaces[curr_edge.face].halfedge = _curr
        if interior:
            self._cfaces[twin_edge.face].halfedge = _twin
            _face_1_idx = new_faces[n_face_idx] #self._new_face(_twin_prev)
            n_face_idx += 1
            self._cfaces[_face_1_idx].halfedge = _twin_prev
            self._chalfedges[_twin_prev].face = _face_1_idx
        
        _face_2_idx = new_faces[n_face_idx]
        n_face_idx += 1 #self._new_face(_next)
        self._cfaces[_face_2_idx].halfedge = _next
        self._chalfedges[_next].face = _face_2_idx

        # Insert the new faces
        _he_0_idx = new_edges[n_edge_idx]
        n_edge_idx += 1
        _he_4_idx = new_edges[n_edge_idx]
        n_edge_idx += 1
        _he_5_idx = new_edges[n_edge_idx]
        n_edge_idx += 1
        
        if interior:
            _he_1_idx = new_edges[n_edge_idx]
            n_edge_idx += 1
            _he_2_idx = new_edges[n_edge_idx]
            n_edge_idx += 1
            _he_3_idx = new_edges[n_edge_idx]
            n_edge_idx += 1

            self._populate_edge(_he_1_idx, _vertex_idx, prev=_twin_next, next=_twin, face=self._chalfedges[_twin].face, twin=_he_2_idx)
            self._populate_edge(_he_2_idx, self._chalfedges[_twin_next].vertex, prev= _he_3_idx, next=_twin_prev, face=_face_1_idx, twin=_he_1_idx)
            self._populate_edge(_he_3_idx,_vertex_idx, prev=_twin_prev, next=_he_2_idx, face=_face_1_idx, twin=_he_4_idx)
        else:
            _he_1_idx = -1
            _he_2_idx = -1
            _he_3_idx = -1
        
        self._populate_edge(_he_0_idx, self._chalfedges[_next].vertex, prev=_curr, next=_prev, face=self._chalfedges[_curr].face, twin=_he_5_idx)
        self._populate_edge(_he_4_idx, self._chalfedges[_curr].vertex, prev=_he_5_idx, next=_next, face=_face_2_idx, twin=_he_3_idx)
        self._populate_edge(_he_5_idx, _vertex_idx, prev=_next, next=_he_4_idx, face=_face_2_idx, twin=_he_0_idx)

        # Update _prev, next
        self._chalfedges[_prev].prev = _he_0_idx
        self._chalfedges[_next].prev = _he_4_idx
        self._chalfedges[_next].next = _he_5_idx

        if interior:
            # Update _twin_next, _twin_prev
            self._chalfedges[_twin_next].next = _he_1_idx
            self._chalfedges[_twin_prev].prev = _he_2_idx
            self._chalfedges[_twin_prev].next = _he_3_idx

            self._chalfedges[_twin].prev = _he_1_idx
        # Update _curr and _twin
        self._chalfedges[_curr].vertex = _vertex_idx
        self._chalfedges[_curr].next = _he_0_idx

        # Update halfedges
        if interior:
            self._cvertices[self._chalfedges[_he_2_idx].vertex].halfedge = _he_1_idx
        
        self._cvertices[self._chalfedges[_prev].vertex].halfedge = _curr
        self._cvertices[self._chalfedges[_he_4_idx].vertex].halfedge = _next
        self._cvertices[_vertex_idx].halfedge = _he_4_idx
        self._cvertices[self._chalfedges[_he_0_idx].vertex].halfedge = _he_5_idx

        if upsample:
            # Make sure these edges emanate from the new vertex stored at _vertex_idx
            if interior:
                self._loop_subdivision_flip_edges.extend([_he_2_idx])
            
            self._loop_subdivision_flip_edges.extend([_he_0_idx])
            self._loop_subdivision_new_vertices.extend([_vertex_idx])
        
        #print(_he_0_idx, _he_1_idx, _he_2_idx, _he_3_idx, _he_4_idx, _he_5_idx)
        #print(_vertex_idx)
        #print(_face_1_idx, _face_2_idx)

        #print('update')
        if live_update:
            if interior:
                #self._update_face_normals([self._chalfedges[_he_0_idx].face, self._chalfedges[_he_1_idx].face, self._chalfedges[_he_2_idx].face, self._chalfedges[_he_4_idx].face])
                #self._update_vertex_neighbors([self._chalfedges[_curr].vertex, self._chalfedges[_twin].vertex, self._chalfedges[_he_0_idx].vertex, self._chalfedges[_he_2_idx].vertex, self._chalfedges[_he_4_idx].vertex])
            
                update_face_normal(self._chalfedges[_he_0_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                update_face_normal(self._chalfedges[_he_1_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                update_face_normal(self._chalfedges[_he_2_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                update_face_normal(self._chalfedges[_he_4_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                
                #print('vertex_neighbours')
                update_single_vertex_neighbours(self._chalfedges[_curr].vertex, self._chalfedges, self._cvertices, self._cfaces)
                #print('n1')
                update_single_vertex_neighbours(self._chalfedges[_twin].vertex, self._chalfedges, self._cvertices, self._cfaces)
                #print('n2')
                update_single_vertex_neighbours(self._chalfedges[_he_0_idx].vertex, self._chalfedges, self._cvertices, self._cfaces)
                #print('n3')
                update_single_vertex_neighbours(self._chalfedges[_he_2_idx].vertex, self._chalfedges, self._cvertices, self._cfaces)
                #print('n')
                update_single_vertex_neighbours(self._chalfedges[_he_4_idx].vertex, self._chalfedges, self._cvertices, self._cfaces)
                #print('vertex_neighbours done')
            
            else:
                #self._update_face_normals([self._chalfedges[_he_0_idx].face, self._chalfedges[_he_4_idx].face])
                #self._update_vertex_neighbors([self._chalfedges[_curr].vertex, self._chalfedges[_prev].vertex, self._chalfedges[_he_0_idx].vertex, self._chalfedges[_he_4_idx].vertex])
                
                update_face_normal(self._chalfedges[_he_0_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                update_face_normal(self._chalfedges[_he_4_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                
                update_single_vertex_neighbours(self._chalfedges[_curr].vertex, self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(self._chalfedges[_prev].vertex, self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(self._chalfedges[_he_0_idx].vertex, self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(self._chalfedges[_he_4_idx].vertex, self._chalfedges, self._cvertices, self._cfaces)
            
            self._clear_flags()
        
        return 1

    cdef int skeleton_split_edges(self, float max_triangle_angle=1.9198622):
        cdef int split_count = 0
        cdef int i
        cdef int n_halfedges = self._halfedges.shape[0]
        cdef float *v0
        cdef float *v1
        cdef float *v2
        cdef float *v3
        cdef float *v4
        cdef float *v5
        cdef float t0d, t1d, t0l, t1l
        cdef float ct0d, ct1d
        cdef int n_edge_idx, n_face_idx, n_vertex_idx

        cdef int* edges_to_split = <int*>PyMem_Malloc(n_halfedges*sizeof(int))
        if not edges_to_split:
            raise MemoryError()
        cdef int* twin_split = <int*>PyMem_Malloc(n_halfedges*sizeof(int))
        if not twin_split:
            raise MemoryError()

        for i in range(n_halfedges):
            twin_split[i] = 0
        
        for i in range(n_halfedges):
            if (not twin_split[i]) and (self._chalfedges[i].vertex != -1):
                v0 = &self._cvertices[self._chalfedges[i].vertex].position0
                v1 = &self._cvertices[self._chalfedges[self._chalfedges[i].next].vertex].position0
                v2 = &self._cvertices[self._chalfedges[self._chalfedges[i].prev].vertex].position0
                t0d = (v0[0]-v1[0])*(v2[0]-v1[0]) + (v0[1]-v1[1])*(v2[1]-v1[1]) + (v0[2]-v1[2])*(v2[2]-v1[2])
                t0l = self._chalfedges[self._chalfedges[i].next].length*self._chalfedges[self._chalfedges[i].prev].length
                
                if t0l > 0:
                    t0d /= t0l
                else:
                    # do not split an edge of length 0
                    continue

                if (self._chalfedges[i].twin != -1):
                    v3 = &self._cvertices[self._chalfedges[self._chalfedges[i].twin].vertex].position0
                    v4 = &self._cvertices[self._chalfedges[self._chalfedges[self._chalfedges[i].twin].next].vertex].position0
                    v5 = &self._cvertices[self._chalfedges[self._chalfedges[self._chalfedges[i].twin].prev].vertex].position0
                
                    t1d = (v3[0]-v4[0])*(v5[0]-v4[0]) + (v3[1]-v4[1])*(v5[1]-v4[1]) + (v3[2]-v4[2])*(v5[2]-v4[2])
                    t1l = self._chalfedges[self._chalfedges[self._chalfedges[i].twin].next].length*self._chalfedges[self._chalfedges[self._chalfedges[i].twin].prev].length

                    if t1l > 0:
                        t1d /= t1l
                    else:
                        # do not split an edge of length 0
                        continue
                else:
                    t1d = 0

                t0d = np.clip(t0d, -1, 1)
                t1d = np.clip(t1d, -1, 1)

                ct0d = np.arccos(t0d)
                ct1d = np.arccos(t1d)

                if (ct1d < max_triangle_angle) or (ct0d < max_triangle_angle):
                    continue

                # split according to the larger of the two triangles
                # (skeleton_edge_split relies on projection from one of the triangles)
                if (ct0d > ct1d):
                    #print(f"Splitting {i}")
                    if self._chalfedges[i].twin != -1:
                        twin_split[self._chalfedges[i].twin] = 1
                    edges_to_split[split_count] = i
                else:
                    #print(f"Splitting {self._chalfedges[i].twin}")
                    twin_split[i] = 1
                    edges_to_split[split_count] = self._chalfedges[i].twin
                split_count += 1
        
        n_edges = self.new_edges(int(split_count*6))
        n_edge_idx = 0
        n_faces = self.new_faces(int(split_count*2))
        n_face_idx = 0
        n_vertices = self.new_vertices(int(split_count))
        n_vertex_idx = 0
        #print(self._halfedges[n_edges])

        for i in range(split_count):
            e = edges_to_split[i]
            #print(i, e, n_edge_idx, n_edges)
            self.skeleton_edge_split(e, 
                                <np.int32_t *> np.PyArray_DATA(n_edges), 
                                <np.int32_t *> np.PyArray_DATA(n_vertices), 
                                <np.int32_t *> np.PyArray_DATA(n_faces), 
                                n_edge_idx, n_vertex_idx, n_face_idx)
            #self.edge_split(e)
            n_edge_idx += 6
            n_face_idx += 2
            n_vertex_idx += 1

        PyMem_Free(edges_to_split)
        PyMem_Free(twin_split)
                
        print('Split count: %d' % (split_count))
        return split_count

    cpdef int skeleton_edge_collapse(self, np.int32_t _curr, bint live_update=1):
        """
        A.k.a. delete two triangles. Remove an edge, defined by halfedge _curr,
        and its associated triangles, while keeping mesh connectivity.

        TODO: A known problem is, in the presence of boundaries, edge_collapse()
        will eventually (after 3-5 remesh steps) produce a few singular edges.

        Parameters
        ----------
            _curr: int
                Pointer to halfedge defining edge to collapse.
            live_update : bool
                Update associated faces and vertices after collapse. Set to 
                False to handle this externally (useful if operating on 
                multiple, disjoint edges).
        """

        cdef halfedge_t *curr_halfedge
        cdef halfedge_t *twin_halfedge
        cdef np.int32_t _prev, _twin, _next, _prev_twin, _prev_twin_vertex, _next_prev_twin, _next_prev_twin_vertex, _twin_next_vertex, _next_twin_twin_next, _next_twin_twin_next_vertex, vl, vd, _dead_vertex, _live_vertex, vn, vtn, face0, face1, face2, face3
        cdef np.int32_t *neighbours_live
        cdef np.int32_t *neighbours_dead
        cdef np.int32_t shared_vertex
        cdef np.float32_t px, py, pz
        cdef bint fast_collapse_bool, interior
        cdef int i, j, twin_count, dead_count
        cdef np.int32_t[5*NEIGHBORSIZE] dead_vertices

        if _curr == -1:
            return 0

        if self._chalfedges[_curr].locally_manifold == 0:
            return 0

        # Create the pointers we need
        curr_halfedge = &self._chalfedges[_curr]
        _prev = curr_halfedge.prev
        _next = curr_halfedge.next
        _twin = curr_halfedge.twin

        if (self._chalfedges[_next].twin == -1) or (self._chalfedges[_prev].twin == -1):
            # Collapsing this edge will create another free edge
            return 0

        interior = (_twin != -1)  # Are we on a boundary?

        if interior:
            #nn = self._vertices['neighbors'][curr_halfedge.vertex]
            #nn_mask = (nn != -1)
            #if (-1 in self._halfedges['twin'][nn]*nn_mask):
            #    return
            
            if not self._check_neighbour_twins(curr_halfedge.vertex):
                return 0

            #nn = self._vertices['neighbors'][self._halfedges['vertex'][_twin]]
            #nn_mask = (nn != -1)
            #if (-1 in self._halfedges['twin'][nn]*nn_mask):
            #    return
            
            if not self._check_neighbour_twins(self._chalfedges[_twin].vertex):
                return 0

            twin_halfedge = &self._chalfedges[_twin]
            _twin_prev = twin_halfedge.prev
            _twin_next = twin_halfedge.next

            if (self._chalfedges[_twin_prev].twin == -1) or (self._chalfedges[_twin_next].twin == -1):
                # Collapsing this edge will create another free edge
                return 0

        _dead_vertex = self._chalfedges[_prev].vertex
        _live_vertex = curr_halfedge.vertex

        # Grab the valences of the 4 points near the edge
        if interior:
            vn, vtn = self._cvertices[self._chalfedges[_next].vertex].valence, self._cvertices[self._chalfedges[_twin_next].vertex].valence

            # Make sure we create no vertices of valence <3 (manifoldness)
            # ((vl + vd - 3) < 4) or 
            #if (vn < 4) or (vtn < 4):
            #    return 0

        vl, vd = self._cvertices[_live_vertex].valence, self._cvertices[_dead_vertex].valence
        
        #if ((vl + vd - 4) < 4):
        #    return 0
        
        cdef bint locally_manifold = self._cvertices[_live_vertex].locally_manifold and self._cvertices[_dead_vertex].locally_manifold

        # Check for creation of multivalent edges and prevent this (manifoldness)
        fast_collapse_bool = (locally_manifold and (vl < NEIGHBORSIZE) and (vd < NEIGHBORSIZE))
        if fast_collapse_bool:
            # Do it the fast way if we can
            neighbours_live = &self._cvertices[_live_vertex].neighbor0
            neighbours_dead = &self._cvertices[_dead_vertex].neighbor0
            twin_count = 0
            shared_vertex = -1
            for i in range(NEIGHBORSIZE):
                if neighbours_live[i] == -1:
                    break
                for j in range(NEIGHBORSIZE):
                    if neighbours_dead[j] == -1:
                        break
                    if self._chalfedges[neighbours_live[i]].vertex == self._chalfedges[neighbours_dead[j]].vertex:
                        if twin_count > 2:
                            break
                        if (twin_count == 0) or ((twin_count > 0) and (self._chalfedges[neighbours_dead[j]].vertex != shared_vertex)):
                            shared_vertex = self._chalfedges[neighbours_live[i]].vertex
                            twin_count += 1
                if twin_count > 2:
                    break

            # no more than two vertices shared by the neighbors of dead and live vertex
            if twin_count != 2:
                return 0

            # assign
            for i in range(NEIGHBORSIZE):
                if (neighbours_dead[i] == -1):
                    continue
                self._chalfedges[self._chalfedges[neighbours_dead[i]].twin].vertex = _live_vertex
                self._chalfedges[self._chalfedges[neighbours_dead[i]].prev].vertex = _live_vertex

            #live_nn = self._vertices['neighbors'][_live_vertex]
            #dead_nn =  self._vertices['neighbors'][_dead_vertex]
            #live_mask = (live_nn != -1)
            #dead_mask = (dead_nn != -1)
            #live_list = self._halfedges['vertex'][live_nn[live_mask]]
            #dead_list = self._halfedges['vertex'][dead_nn[dead_mask]]
        else:
            # grab the set of halfedges pointing to dead_vertices
            
            dead_count = 0
            for i in range(self._halfedges.shape[0]):
                if self._chalfedges[i].vertex == _dead_vertex:
                    dead_vertices[dead_count] = i
                    dead_count += 1
                    if dead_count > 5*NEIGHBORSIZE:
                        print(f'WARNING: Way too many dead vertices: {dead_count}! Politely declining to collapse.')
                        return 0

            # loop over all live vertices and check for twins in dead_vertices,
            # as we do in fast_collapse
            twin_count = 0
            shared_vertex = -1
            for i in range(self._halfedges.shape[0]):
                if self._chalfedges[i].twin == -1:
                    continue
                if self._chalfedges[i].vertex == _live_vertex:
                    for j in range(dead_count):
                        if self._chalfedges[dead_vertices[j]].twin == -1:
                            continue
                        if self._chalfedges[self._chalfedges[i].twin].vertex == self._chalfedges[self._chalfedges[dead_vertices[j]].twin].vertex:
                            if twin_count > 2:
                                break
                            if (twin_count == 0) or ((twin_count > 0) and (self._chalfedges[self._chalfedges[dead_vertices[j]].twin].vertex != shared_vertex)):
                                shared_vertex = self._chalfedges[self._chalfedges[i].twin].vertex
                                twin_count += 1
                    if twin_count > 2:
                        break

            # no more than two vertices shared by the neighbors of dead and live vertex
            if twin_count != 2:
                return 0

            # assign
            for i in range(dead_count):
                self._chalfedges[dead_vertices[i]].vertex = _live_vertex

            #twin_mask = (self._halfedges['twin'] != -1)
            #dead_mask = (self._halfedges['vertex'] == _dead_vertex)
            #dead_list = self._halfedges['vertex'][self._halfedges['twin'][dead_mask & twin_mask]]
            #live_list = self._halfedges['vertex'][self._halfedges['twin'][(self._halfedges['vertex'] == _live_vertex) & twin_mask]]

        # twin_list = list((set(dead_list) & set(live_list)) - set([-1]))
        # if len(twin_list) != 2:
        #     return 0
            
        # Collapse to the midpoint of the original edge vertices
        # if fast_collapse_bool:
        #    self._halfedges['vertex'][self._halfedges['twin'][dead_nn[dead_mask]]] = _live_vertex
        # else:
        #     self._halfedges['vertex'][dead_mask] = _live_vertex
        
        # if fast_collapse_bool:
        #     if not self._check_collapse_fast(_live_vertex, _dead_vertex):
        #         return
        # else:
        #     if not self._check_collapse_slow(_live_vertex, _dead_vertex):
        #         return
        
        # _live_pos = self._vertices['position'][_live_vertex]
        # _dead_pos = self._vertices['position'][_dead_vertex]
        # self._vertices['position'][_live_vertex] = 0.5*(_live_pos + _dead_pos)

        #px = 0.5*(self._cvertices[_live_vertex].position0 + self._cvertices[_dead_vertex].position0)
        #py = 0.5*(self._cvertices[_live_vertex].position1 + self._cvertices[_dead_vertex].position1)
        #pz = 0.5*(self._cvertices[_live_vertex].position2 + self._cvertices[_dead_vertex].position2)
        self._cvertices[_live_vertex].position0 = self._cvertices[_live_vertex].position0
        self._cvertices[_live_vertex].position1 = self._cvertices[_live_vertex].position1
        self._cvertices[_live_vertex].position2 = self._cvertices[_live_vertex].position2
        
        # update valence of vertex we keep
        self._cvertices[_live_vertex].valence = vl + vd - 3
        
        # delete dead vertex
        self._vertices[_dead_vertex] = -1
        self._vertex_vacancies.append(_dead_vertex)

        # Zipper the remaining triangles
        self._zipper(_next, _prev)
        if interior:
            self._zipper(_twin_next, _twin_prev)
        # We need some more pointers
        # TODO: make these safer
        _prev_twin = self._chalfedges[_prev].twin
        _prev_twin_vertex = self._chalfedges[_prev_twin].vertex
        _next_prev_twin = self._chalfedges[_prev_twin].next
        _next_prev_twin_vertex = self._chalfedges[_next_prev_twin].vertex
        if interior:
            _twin_next_vertex = self._chalfedges[_twin_next].vertex
            _next_twin_twin_next = self._chalfedges[self._chalfedges[_twin_next].twin].next
            _next_twin_twin_next_vertex = self._chalfedges[_next_twin_twin_next].vertex
            
        # Make sure we have good _vertex_halfedges references
        self._cvertices[_live_vertex].halfedge = _prev_twin
        self._cvertices[_prev_twin_vertex].halfedge = _next_prev_twin
        if interior:
            self._cvertices[_twin_next_vertex].halfedge = self._chalfedges[_twin_next].twin
            self._cvertices[_next_twin_twin_next_vertex].halfedge = self._chalfedges[_next_twin_twin_next].next

        # Grab faces to update
        face0 = self._chalfedges[self._chalfedges[_next].twin].face
        face1 = self._chalfedges[self._chalfedges[_prev].twin].face
        if interior:
            face2 = self._chalfedges[self._chalfedges[_twin_next].twin].face
            face3 = self._chalfedges[self._chalfedges[_twin_prev].twin].face

        # Delete the inner triangles
        self._face_delete(_curr)
        if interior:
            self._face_delete(_twin)

        try:
            if live_update:
                if interior:
                    # Update faces
                    #self._update_face_normals([face0, face1, face2, face3])
                    #self._update_vertex_neighbors([_live_vertex, _prev_twin_vertex, _next_prev_twin_vertex, _twin_next_vertex])
                    
                    update_face_normal(face0, self._chalfedges, self._cvertices, self._cfaces)
                    update_face_normal(face1, self._chalfedges, self._cvertices, self._cfaces)
                    update_face_normal(face2, self._chalfedges, self._cvertices, self._cfaces)
                    update_face_normal(face3, self._chalfedges, self._cvertices, self._cfaces)
                    
                    update_single_vertex_neighbours(_live_vertex, self._chalfedges, self._cvertices, self._cfaces)
                    update_single_vertex_neighbours(_prev_twin_vertex, self._chalfedges, self._cvertices, self._cfaces)
                    update_single_vertex_neighbours(_next_prev_twin_vertex, self._chalfedges, self._cvertices, self._cfaces)
                    update_single_vertex_neighbours(_twin_next_vertex, self._chalfedges, self._cvertices, self._cfaces)
                    
                else:
                    #self._update_face_normals([face0, face1])
                    #self._update_vertex_neighbors([_live_vertex, _prev_twin_vertex, _next_prev_twin_vertex])
                    
                    update_face_normal(face0, self._chalfedges, self._cvertices, self._cfaces)
                    update_face_normal(face1, self._chalfedges, self._cvertices, self._cfaces)
                    
                    update_single_vertex_neighbours(_live_vertex, self._chalfedges, self._cvertices, self._cfaces)
                    update_single_vertex_neighbours(_prev_twin_vertex, self._chalfedges, self._cvertices, self._cfaces)
                    update_single_vertex_neighbours(_next_prev_twin_vertex, self._chalfedges, self._cvertices, self._cfaces)
                    
                self._clear_flags()
        
        except RuntimeError as e:
            print(_curr, _twin, _next, _prev, _twin_next, _twin_prev, _next_prev_twin, _next_twin_twin_next, _prev_twin)
            print([_live_vertex, _prev_twin_vertex, _next_prev_twin_vertex, _twin_next_vertex])
            raise e
        
        return 1

    def skeleton_collapse_edges(self, float collapse_threshold):
        cdef int collapse_count = 0
        cdef int collapse_fails = 0
        cdef int i
        cdef int n_halfedges = self._halfedges.shape[0]
        cdef float *n1
        cdef float *n2
        cdef float nd
        
        #find which vertices are locally manifold
        self._update_vertex_locally_manifold()
        
        for i in range(n_halfedges):
            if (self._chalfedges[i].vertex != -1) and (self._chalfedges[i].length < collapse_threshold):
                collapse_ret = self.skeleton_edge_collapse(i)
                collapse_count += collapse_ret
                collapse_fails += (1-collapse_ret)
        print('Collapse count: ' + str(collapse_count) + '[' + str(collapse_fails) +' failed]')
        
        return collapse_count

    cdef int skeleton_remesh(self, float target_edge_length=-1, float max_triangle_angle=1.9198622):
        cdef int k, i, ct
        cdef float collapse_threshold, xl, xu, yl, yu, zl, zu, diag  #, split_threshold

        cdef int n_halfedges = self._halfedges.shape[0]
        cdef int n_vertices = self._vertices.shape[0]

        if (target_edge_length < 0):
            # Guess edge_length
            xl, yl, zl, xu, yu, zu = self.bbox
            diag = math.sqrt((xu-xl)*(xu-xl)+(yu-yl)*(yu-yl)+(zu-zl)*(zu-zl))
            collapse_threshold = 0.002*diag
        else:
            collapse_threshold = target_edge_length

        #split_threshold = 1.66*collapse_threshold

        self._update_singular_edge_locally_manifold()
        self._update_singular_vertex_locally_manifold()
        
        print(f"Target edge length: {collapse_threshold}")
        ct = self.skeleton_collapse_edges(collapse_threshold)

        self._update_singular_edge_locally_manifold()
        self._update_singular_vertex_locally_manifold()

        ct = self.skeleton_split_edges(max_triangle_angle=max_triangle_angle)
        # ct = self.split_edges(split_threshold)

        # Let's double-check the mesh manifoldness
        self._manifold = 0
        self.manifold
        self._initialize_curvature_vectors()

        return 1

    def opt_skeleton(self, points, sigma, max_iter=10, lam=[0], target_edge_length=-1, **kwargs):
        from ch_shrinkwrap.conj_grad import SkeletonConjGrad

        self.remesh_frequency = 1
        rf = 1  # manually set remesh frequency to 1

        # initialize area values (used in termination condition)
        original_area = self.area()
        last_area, area = original_area, 0
        area_variation_factor = kwargs.get('area_variation_factor')
        max_triangle_angle = PI*kwargs.get('max_triangle_angle')/180.0

        # initialize skeleton, construct Voronoi diagram once
        n = self._halfedges['vertex'][self._vertices['neighbors']]
        n[self._vertices['neighbors'] == -1] = -1
        cg = SkeletonConjGrad(self._vertices['position'], self._vertices['normal'], n, mesh=self)
        
        # Loop over iterations
        for j in range(max_iter//rf):
            k = (self._vertices['halfedge'] != -1)
            print(f"{k.sum()} vertices")

            n = self._halfedges['vertex'][self._vertices['neighbors']]
            n[self._vertices['neighbors'] == -1] = -1

            # Update positions
            cg.vertex_neighbors, cg.vertex_normals, cg.vertices = n, self._vertices['normal'], self._vertices['position']

            vp = cg.search(np.zeros_like(self._vertices['position']),lams=lam,num_iters=rf)

            self._vertices['position'][k] = vp[k]

            # self._faces['normal'][:] = -1
            # self._vertices['neighbors'][:] = -1
            self._face_normals_valid = 0
            self._vertex_normals_valid = 0
            self.face_normals
            self.vertex_neighbors

            # Remesh
            if ((((j+1)*rf) % self.remesh_frequency) == 0):
                # print(f"Max angle: {max_triangle_angle}")
                # self.skeleton_remesh(target_edge_length=target_edge_length,
                #                      max_triangle_angle=max_triangle_angle)
                self.remesh(target_edge_length=target_edge_length)

            # Check if area ratio is met
            area = self.area()
            area_ratio = math.fabs(last_area-area)/original_area
            print(f"Area ratio is {area_ratio:.4f}")
            if area_ratio < area_variation_factor:
                break
            last_area = area
        
    # def shrink_wrap(self, points=None, sigma=None, method='conjugate_gradient', max_iter=None, **kwargs):

    #     if method not in DESCENT_METHODS:
    #         print('Unknown gradient descent method. Using {}.'.format(DEFAULT_DESCENT_METHOD))
    #         method = DEFAULT_DESCENT_METHOD

    #     if max_iter is None:
    #         max_iter = self.max_iter
        
    #     # save points and sigmas so we can call again to continue
    #     if points is None:
    #         points = self._points

    #     if sigma is None:
    #         sigma = self._sigma
        
    #     opts = dict(points=points,
    #                 sigma=sigma, 
    #                 max_iter=max_iter,
    #                 step_size=self.step_size, 
    #                 beta_1=self.beta_1, 
    #                 beta_2=self.beta_2,
    #                 eps=self.eps,
    #                 **kwargs)

    #     #opts.update(kwargs)

    #     self._points = points
    #     self._sigma = sigma

    #     return getattr(self, 'opt_{}'.format(method))(**opts)

    #     # if method == 'euler':
    #     #     return self.opt_euler(**opts)
    #     # elif method == 'expectation_maximization':
    #     #     return self.opt_expectation_maximization(**opts)
    #     # elif method == 'adam':
    #     #     return self.opt_adam(**opts)
