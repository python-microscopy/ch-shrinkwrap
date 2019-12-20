import numpy as np

from PYME.experimental._triangle_mesh import TriangleMesh

import membrane_mesh_utils

# Gradient descent methods
DESCENT_METHODS = ['euler', 'expectation_maximization', 'adam']

class MembraneMesh(TriangleMesh):
    def __init__(self, vertices=None, faces=None, mesh=None, **kwargs):
        super(MembraneMesh, self).__init__(vertices, faces, mesh, **kwargs)

        # Bending stiffness coefficients (in units of kbT)
        self.kc = 0.1
        self.kg = -0.1

        # Gradient weight
        self.a = 1.
        self.c = -1.

        # Adam optimizer parameters
        self.step_size = 1
        self.beta_1 = 0.8
        self.beta_2 = 0.7
        self.eps = 1e-8
        self.max_iter = 250

        # Coloring info
        self._H = None
        self._K = None
        self._E = None
        self._pE = None
        
        self.vertex_properties.extend(['E', 'pE'])

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def E(self):
        if self._E is None:
            self.curvature_grad()
        self._E[np.isnan(self._E)] = 0
        return self._E

    @property
    def H(self):
        if self._H is None:
            self.curvature_grad()
        self._H[np.isnan(self._H)] = 0
        return self._H

    @property
    def K(self):
        if self._K is None:
            self.curvature_grad()
        self._K[np.isnan(self._K)] = 0
        return self._K

    @property
    def pE(self):
        if self._pE is None:
            self.curvature_grad()
            self._pE[np.isnan(self._pE)] = 0
        return self._pE

    @property
    def _mean_edge_length(self):
        return np.mean(self._halfedges['length'][self._halfedges['length'] != -1])

    def remesh(self, n=5, target_edge_length=-1, l=0.5, n_relax=10):
        super(MembraneMesh, self).remesh(n, target_edge_length, l, n_relax)
        # Reset H, E, K values
        self._H = None
        self._K = None
        self._E = None

    def _compute_curvature_tensor_eig(self, Mvi):
        """
        Return the first two eigenvalues and eigenvectors of 3x3 curvature 
        tensor. The third eigenvector is the unit normal of the point for
        which the curvature tensor is defined.

        This is a closed-form solution, and it assumes no eigenvalue is 0.

        Parameters
        ----------
            Mvi : np.array
                3x3 curvature tensor at a point.

        Returns
        -------
            l1, l2 : float
                Eigenvalues
            v1, v2 : np.array
                Eigenvectors
        """
        # Solve the eigenproblem in closed form
        m00 = Mvi[0,0]
        m01 = Mvi[0,1]
        m02 = Mvi[0,2]
        m11 = Mvi[1,1]
        m12 = Mvi[1,2]
        m22 = Mvi[2,2]

        # Here we use the fact that Mvi is symnmetric and we know
        # one of the eigenvalues must be 0
        p = -m00*m11 - m00*m22 + m01*m01 + m02*m02 - m11*m22 + m12*m12
        q = m00 + m11 + m22
        r = np.sqrt(4*p + q*q)
        
        # Eigenvalues
        l1 = 0.5*(q-r)
        l2 = 0.5*(q+r)

        def safe_divide(x, y):
            if y == 0:
                return 0
            return 1.*x/y

        # Now calculate the eigenvectors, assuming x = 1
        z1n = ((m00 - l1)*(m11 - l1) - (m01*m01))
        z1d = (m01*m12 - m02*(m11 - l1))
        z1 = safe_divide(z1n, z1d)
        y1n = (m12*z1 + m01)
        y1d = (m11 - l1)
        y1 = safe_divide(y1n, y1d)
        
        v1 = np.array([1., y1, z1])
        v1_norm = np.sqrt((v1*v1).sum())
        v1 = v1/v1_norm
        
        z2n = ((m00 - l2)*(m11 - l2) - (m01*m01))
        z2d = (m01*m12 - m02*(m11 - l2))
        z2 = safe_divide(z2n, z2d)
        y2n = (m12*z2 + m01)
        y2d = (m11 - l2)
        y2 = safe_divide(y2n, y2d)
        
        v2 = np.array([1., y2, z2])
        v2_norm = np.sqrt((v2*v2).sum())
        v2 = v2/v2_norm

        return l1, l2, v1, v2

    def curvature_grad(self, dN=0.1):
        """
        Estimate curvature. Here we follow a mix of ESTIMATING THE 
        TENSOR OF CURVATURE OF A SURFACE FROM A POLYHEDRAL 
        APPROXIMATION by Gabriel Taubin from Proceedings of IEEE 
        International Conference on Computer Vision, June 1995 and 
        Estimating the PrincipalCurvatures and the Darboux Frame 
        From Real 3-D Range Data by Eyal Hameiri and Ilan Shimshon 
        from IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS-PART
        B: CYBERNETICS, VOL. 33, NO. 4, AUGUST 2003
        """
        H = np.zeros(self._vertices.shape[0])
        K = np.zeros(self._vertices.shape[0])
        dH = np.zeros(self._vertices.shape[0])
        dK = np.zeros(self._vertices.shape[0])
        dE_neighbors = np.zeros(self._vertices.shape[0])
        I = np.eye(3)
        areas = np.zeros(self._vertices.shape[0])
        for iv in range(self._vertices.shape[0]):
            if self._vertices['halfedge'][iv] == -1:
                continue

            # Vertex and its normal
            vi = self._vertices['position'][iv,:]
            Nvi = self._vertices['normal'][iv,:]

            p = I - Nvi[:,None]*Nvi[None,:] # np.outer(Nvi, Nvi)
                
            # vertex nearest neighbors
            neighbors = self._vertices['neighbors'][iv]
            neighbor_mask = (neighbors != -1)
            neighbor_vertices = self._halfedges['vertex'][neighbors]
            vjs = self._vertices['position'][neighbor_vertices[neighbor_mask]]
            Nvjs = self._vertices['normal'][neighbor_vertices[neighbor_mask]]

            # Neighbor vectors & displaced neighbor tangents
            dvs = vjs - vi[None,:]
            dvs_1 = dvs - (Nvi*dN)[None, :]

            # radial weighting
            r_sum = np.sum(1./np.sqrt((dvs*dvs).sum(1)))

            # Norms
            dvs_norm = np.sqrt((dvs*dvs).sum(1))
            dvs_1_norm = np.sqrt((dvs_1*dvs_1).sum(1))

            # Hats
            dvs_hat = dvs/dvs_norm[:,None]
            dvs_1_hat = dvs_1/dvs_1_norm[:,None]

            # Tangents
            T_thetas = np.dot(p,-dvs.T).T
            Tijs = T_thetas/np.sqrt((T_thetas*T_thetas).sum(1)[:,None])
            Tijs[np.sum(T_thetas,axis=1) == 0, :] = 0

            # Edge normals subtracted from vertex normals
            Ni_diffs = np.sqrt(2. - 2.*np.sqrt(1.-((Nvi[None,:]*dvs_hat).sum(1))**2))
            Nj_diffs = np.sqrt(2. - 2.*np.sqrt(1.-((Nvjs*dvs_hat).sum(1))**2))
            Nj_1_diffs = np.sqrt(2. - 2.*np.sqrt(1.-((Nvjs*dvs_1_hat).sum(1))**2))

            kjs = 2.*Nj_diffs/dvs_norm
            kjs_1 = 2.*Nj_1_diffs/dvs_1_norm

            k = 2.*np.sign((Nvi[None,:]*dvs).sum(1))*Ni_diffs/dvs_norm
            w = (1./dvs_norm)/r_sum

            # Calculate areas
            Aj = self._faces['area'][self._halfedges['face'][neighbors[neighbor_mask]]]
            areas[iv] = np.sum(Aj)

            dEj = Aj*w*2.*self.kc*(kjs_1**2 - kjs**2)/dN

            Mvi = (w[None,:,None]*k[None,:,None]*Tijs.T[:,:,None]*Tijs[None,:,:]).sum(axis=1)

            l1, l2, v1, v2 = self._compute_curvature_tensor_eig(Mvi)

            # Eigenvectors
            m = np.vstack([v1, v2, Nvi]).T

            # Principal curvatures
            k_1 = 3.*l1 - l2 #e[0] - e[1]
            k_2 = 3.*l2 - l1 #e[1] - e[0]

            # Mean and Gaussian curvatures
            H[iv] = 0.5*(k_1 + k_2)
            K[iv] = k_1*k_2

            # Now calculate the shift
            # We construct a quadratic in the space of T_1 vs. T_2
            t_1, t_2, _ = np.dot(vjs-vi,m).T
            A = np.array([t_1**2, t_2**2]).T
            
            # Update the equation y-intercept to displace the curve along
            # the normal direction
            b = np.dot(A,np.array([k_1,k_2])) - dN
            
            # Solve
            # Previously k_p, _, _, _ = np.linalg.lstsq(A, b)
            k_p = np.dot(np.dot(np.linalg.pinv(np.dot(A.T,A)),A.T),b) 

            # Finite differences of displaced curve and original curve
            dH[iv] = (0.5*(k_p[0] + k_p[1]) - H[iv])/dN
            dK[iv] = ((k_p[0]-k_1)*k_2 + k_1*(k_p[1]-k_2))/dN

            dE_neighbors[iv] = np.sum(dEj)

        # Calculate Canham-Helfrich energy functional
        E = areas*(2.*self.kc*H**2 + self.kg*K)

        self._H = H
        self._E = E
        self._K = K
        
        #pEi = np.exp(-250.*E)
        pEi = np.exp(-4.*E)
        #self._pE = (1./np.median(pEi))*pEi
        self._pE = pEi
        ## Take into account the change in neighboring energies for each
        # vertex shift
        dEdN = (areas*(4.*self.kc*H*dH + self.kg*dK) + dE_neighbors)*(1.-self._pE)
        # dEdN = -(4.*self.kc*H*dH + self.kg*dK)*pE
        # 250 = 1/kbT where kb in nm
        # dpdN = -250.*np.exp(-250.*E)*dEdN
        
        # Return derivative of Boltzmann distribution
        return -dEdN[:,None]*self._vertices['normal']

    def point_attraction_grad(self, points, sigma, w=0.95):
        """
        Attractive force of membrane to points.

        Parameters
        ----------
            points : np.array
                3D point cloud to fit.
            sigma : float
                Localization uncertainty of points.
        """
        dirs = []

        # pt_cnt_dist_2 will eventually be a MxN (# points x # vertices) matrix, but becomes so in
        # first loop iteration when we add a matrix to this scalar
        # pt_cnt_dist_2 = 0

        # for j in range(points.shape[1]):
        #     pt_cnt_dist_2 = pt_cnt_dist_2 + (points[:,j][:,None] - self._vertices['position'][:,j][None,:])**2

        charge_sigma = self._mean_edge_length/2.5
        # pt_weight_matrix = 1. - w*np.exp(-pt_cnt_dist_2/(2*charge_sigma**2))
        pt_weight_matrix = np.zeros((points.shape[0], self._vertices.shape[0]), 'f4')
        membrane_mesh_utils.calculate_pt_cnt_dist_2(points, self._vertices, pt_weight_matrix, w, charge_sigma)
        pt_weights = np.prod(pt_weight_matrix, axis=1)
        for i in range(self._vertices.shape[0]): 
            if self._vertices['halfedge'][i] != -1:
                d = self._vertices['position'][i, :] - points
                dd = (d*d).sum(1)
                
                r = np.sqrt(dd)/sigma
                
                rf = -(1-r**2)*np.exp(-r**2/2) + (1-np.exp(-(r-1)**2/2))*(r/(r**3 + 1))

                # Points at the vertex we're interested in are not de-weighted by the
                # pt_weight_matrix
                rf = rf*(pt_weights/pt_weight_matrix[:, i])
                
                attraction = (-d*(rf/np.sqrt(dd))[:,None]).sum(0)
            else:
                attraction = np.array([0,0,0])
            
            dirs.append(attraction)

        dirs = np.vstack(dirs)
        dirs[self._vertices['halfedge'] == -1] = 0

        return dirs

    def point_attraction_grad_kdtree(self, points, sigma, w=0.95, search_k=200):
        """
        Attractive force of membrane to points.

        Parameters
        ----------
            points : np.array
                3D point cloud to fit.
            sigma : float
                Localization uncertainty of points.
            w : float
                Weight
            search_r : float
                Max distance of points from vertex to consider
        """
        import scipy.spatial

        dirs = []

        # pt_cnt_dist_2 will eventually be a MxN (# points x # vertices) matrix, but becomes so in
        # first loop iteration when we add a matrix to this scalar
        # pt_cnt_dist_2 = 0

        # for j in range(points.shape[1]):
        #     pt_cnt_dist_2 = pt_cnt_dist_2 + (points[:,j][:,None] - self._vertices['position'][:,j][None,:])**2

        charge_sigma = self._mean_edge_length/2.5
        charge_var = (2*charge_sigma**2)

        # pt_weight_matrix = 1. - w*np.exp(-pt_cnt_dist_2/(2*charge_sigma**2))
        # pt_weights = np.prod(pt_weight_matrix, axis=1)

        # Compute a KDTree on points
        tree = scipy.spatial.cKDTree(points)

        for i in range(self._vertices.shape[0]):
            if self._vertices['halfedge'][i] != -1:
                _, neighbors = tree.query(self._vertices['position'][i,:], search_k)
                # neighbors = tree.query_ball_point(self._vertices['position'][i,:], search_r)
                try:
                    d = self._vertices['position'][i,:] - points[neighbors]
                except(IndexError):
                    print('whaaa?')
                    print(i, neighbors)
                dd = (d*d).sum(1)
                pt_weight_matrix = 1. - w*np.exp(-dd/charge_var)
                pt_weights = np.prod(pt_weight_matrix)
                r = np.sqrt(dd)/sigma[neighbors]
                
                rf = -(1-r**2)*np.exp(-r**2/2) + (1-np.exp(-(r-1)**2/2))*(r/(r**3 + 1))

                # Points at the vertex we're interested in are not de-weighted by the
                # pt_weight_matrix
                rf = rf*(pt_weights/pt_weight_matrix)
                
                attraction = (-d*(rf/np.sqrt(dd))[:,None]).sum(0)
            else:
                attraction = np.array([0,0,0])
            
            dirs.append(attraction)

        dirs = np.vstack(dirs)
        dirs[self._vertices['halfedge'] == -1] = 0

        return dirs

    def grad(self, points, sigma):
        """
        Gradient between points and the surface.

        Parameters
        ----------
            points : np.array
                3D point cloud to fit.
            sigma : float
                Localization uncertainty of points.
        """
        dN = 0.1
        curvature = self.curvature_grad(dN=dN)
        attraction = self.point_attraction_grad_kdtree(points, sigma)

        # ratio = np.nanmean(np.linalg.norm(curvature,axis=1)/np.linalg.norm(attraction,axis=1))
        # print('Ratio: ' + str(ratio))

        g = self.a*attraction + self.c*curvature
        return g

    def opt_adam(self, points, sigma, max_iter=250, step_size=1, beta_1=0.9, beta_2=0.999, eps=1e-8, **kwargs):
        """
        Performs Adam optimization (https://arxiv.org/abs/1412.6980) on
        fit of surface mesh surf to point cloud points.

        Parameters
        ----------
            points : np.array
                3D point cloud to fit.
            sigma : float
                Localization uncertainty of points.
        """
        # Initialize moment vectors
        m = np.zeros(self._vertices['position'].shape)
        v = np.zeros(self._vertices['position'].shape)

        t = 0
        # g_mag_prev = 0
        # g_mag = 0
        while (t < max_iter):
            print('Iteration %d ...' % t)
            
            t += 1
            # Gaussian noise std
            noise_sigma = np.sqrt(self.step_size / ((1 + t)**0.55))
            # Gaussian noise
            noise = np.random.normal(0, noise_sigma, self._vertices['position'].shape)
            # Calculate graident for each point on the  surface, 
            g = self.grad(points, sigma)
            # add Gaussian noise to the gradient
            g += noise
            # Update first biased moment 
            m = beta_1 * m + (1. - beta_1) * g
            # Update second biased moment
            v = beta_2 * v + (1. - beta_2) * np.multiply(g, g)
            # Remove biases on moments & calculate update weight
            a = step_size * np.sqrt(1. - beta_2**t) / (1. - beta_1**t)
            # Update the surface
            self._vertices['position'] += a * m / (np.sqrt(v) + eps)

    def opt_euler(self, points, sigma, max_iter=100, step_size=1, eps=0.00001, **kwargs):
        """
        Normal gradient descent.

        Parameters
        ----------
            points : np.array
                3D point cloud to fit.
            sigma : float
                Localization uncertainty of points.
        """

        # Calculate target lengths for remesh steps
        # initial_length = self._mean_edge_length
        # final_length = np.max(sigma)
        # m = (final_length - initial_length)/max_iter
        
        for _i in np.arange(max_iter):

            print('Iteration %d ...' % _i)
            
            # Calculate the weighted gradient
            shift = step_size*self.grad(points, sigma)

            # Update the vertices
            self._vertices['position'] += shift

            self._faces['normal'][:] = -1
            self._vertices['neighbors'][:] = -1
            self.face_normals
            self.vertex_neighbors

            # If we've reached precision, terminate
            if np.all(shift < eps):
                return

            # # Remesh
            # if (np.mod(_i, 19) == 0) and (_i != 0):
            #     target_length = initial_length + m*_i
            #     print('Target length: ' + str(target_length))
            #     self.remesh(5, target_length, 0.5, 10)
            #     print('Mean length: ' + str(self._mean_edge_length))

    def opt_expectation_maximization(self, points, sigma, max_iter=100, step_size=1, eps=0.00001, **kwargs):
        for _i in np.arange(max_iter):

            print('Iteration %d ...' % _i)

            if _i % 2:
                dN = 0.1
                grad = self.c*self.curvature_grad(dN=dN)
            else:
                grad = self.a*self.point_attraction_grad_kdtree(points, sigma)

            # Calculate the weighted gradient
            shift = step_size*grad

            # Update the vertices
            self._vertices['position'] += shift

            self._faces['normal'][:] = -1
            self._vertices['neighbors'][:] = -1
            self.face_normals
            self.vertex_neighbors

            # If we've reached precision, terminate
            if np.all(shift < eps):
                return

    def shrink_wrap(self, points, sigma, method='euler'):

        if method not in DESCENT_METHODS:
            print('Unknown gradient descent method. Using default.')
            method = 'euler'

        opts = dict(points=points, 
                    sigma=sigma, 
                    max_iter=self.max_iter, 
                    step_size=self.step_size, 
                    beta_1=self.beta_1, 
                    beta_2=self.beta_2,
                    eps=self.eps)

        if method == 'euler':
            return self.opt_euler(**opts)
        elif method == 'expectation_maximization':
            return self.opt_expectation_maximization(**opts)
        elif method == 'adam':
            return self.opt_adam(**opts)
