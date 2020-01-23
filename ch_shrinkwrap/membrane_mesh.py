import numpy as np

from PYME.experimental._triangle_mesh import TriangleMesh

from ch_shrinkwrap import membrane_mesh_utils

# Gradient descent methods
DESCENT_METHODS = ['euler', 'expectation_maximization', 'adam']
DEFAULT_DESCENT_METHOD = 'euler'

KB = 8.617e-5  # Boltzmann constant (eV/K)
H = 4.135e-15  # eV*s

class MembraneMesh(TriangleMesh):
    def __init__(self, vertices=None, faces=None, mesh=None, **kwargs):
        super(MembraneMesh, self).__init__(vertices, faces, mesh, **kwargs)

        self.temp = 25  # Celsius

        # Bending stiffness coefficients
        self.kc = 0.514  # eV  (roughly 20 kbt at 25C)
        self.kg = 0.0  # eV

        # Spotaneous curvature
        # Keep in mind the curvature convention we're using. Into the surface is
        # "down" and positive curvature bends "up".
        self.c0 = 0.0 # -0.02  # nm^{-1} for a sphere of radius 50

        # Optimizer parameters
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
        self._rf = None

        self.vertex_properties.extend(['E', 'pE', 'rf'])

        # Number of neighbors to use in self.point_attraction_grad_kdtree
        self.search_k = 200

        # Percentage of vertices to skip on each refinement iteration
        self.skip_prob = 0.0

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.kbt = KB*(self.temp + 273.15)  # eV

        # Curvature probability partition function
        # approximated as \sum_{j=1}^\infty e^{-\frac{j}{self.kbt}}
        # self._Q = 1.0/((np.exp(1)**(1.0/self.kbt))-1.0)  # unitless
        # self._Q = 2.0*self.kbt*np.sinh(self.kc/self.kbt)
        # self._Q = (self.kbt - self.kbt*np.exp(-self.kc/self.kbt))
        self._Q = np.sqrt(np.pi/2.0)/np.sqrt(self.kc/self.kbt)

    @property
    def E(self):
        if self._E is None:
            self.curvature_prob()
        self._E[np.isnan(self._E)] = 0
        return self._E

    @property
    def H(self):
        if self._H is None:
            self.curvature_prob()
        self._H[np.isnan(self._H)] = 0
        return self._H

    @property
    def K(self):
        if self._K is None:
            self.curvature_prob()
        self._K[np.isnan(self._K)] = 0
        return self._K

    @property
    def pE(self):
        if self._pE is None:
            self.curvature_prob()
            self._pE[np.isnan(self._pE)] = 0
        return self._pE

    @property
    def rf(self):
        return self._rf

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

    def curvature_prob(self, dN=0.1, skip_prob=0.0):
        """
        Estimate curvature. Here we follow a mix of ESTIMATING THE 
        TENSOR OF CURVATURE OF A SURFACE FROM A POLYHEDRAL 
        APPROXIMATION by Gabriel Taubin from Proceedings of IEEE 
        International Conference on Computer Vision, June 1995 and 
        Estimating the PrincipalCurvatures and the Darboux Frame 
        From Real 3-D Range Data by Eyal Hameiri and Ilan Shimshon 
        from IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS-PART
        B: CYBERNETICS, VOL. 33, NO. 4, AUGUST 2003

        Units of energy are in kg*nm^2/s^2
        """
        H = np.zeros(self._vertices.shape[0])
        K = np.zeros(self._vertices.shape[0])
        dH = np.zeros(self._vertices.shape[0])
        dK = np.zeros(self._vertices.shape[0])
        dE_neighbors = np.zeros(self._vertices.shape[0])
        I = np.eye(3)
        areas = np.zeros(self._vertices.shape[0])
        skip = np.random.rand(self._vertices.shape[0])
        for iv in range(self._vertices.shape[0]):
            if self._vertices['halfedge'][iv] == -1:
                continue

            # Monte carlo selection of vertices to update
            # Stochastically choose which vertices to adjust
            if skip[iv] < skip_prob:
                continue

            # Vertex and its normal
            vi = self._vertices['position'][iv,:]  # nm
            Nvi = self._vertices['normal'][iv,:]  # unitless

            p = I - Nvi[:,None]*Nvi[None,:]  # unitless
                
            # vertex nearest neighbors
            neighbors = self._vertices['neighbors'][iv]
            neighbor_mask = (neighbors != -1)
            neighbor_vertices = self._halfedges['vertex'][neighbors]
            vjs = self._vertices['position'][neighbor_vertices[neighbor_mask]]  # nm
            Nvjs = self._vertices['normal'][neighbor_vertices[neighbor_mask]]  # unitless

            # Neighbor vectors & displaced neighbor tangents
            dvs = vjs - vi[None,:]  # nm
            dvs_1 = dvs - (Nvi*dN)[None, :]  # nm

            # radial weighting
            r_sum = np.sum(1./np.sqrt((dvs*dvs).sum(1)))  # 1/nm

            # Norms
            dvs_norm = np.sqrt((dvs*dvs).sum(1))  # nm
            dvs_1_norm = np.sqrt((dvs_1*dvs_1).sum(1))  # nm

            # Hats
            dvs_hat = dvs/dvs_norm[:,None]  # unitless
            dvs_1_hat = dvs_1/dvs_1_norm[:,None]  # unitless

            # Tangents
            T_thetas = np.dot(p,-dvs.T).T  # nm^2
            Tijs = T_thetas/np.sqrt((T_thetas*T_thetas).sum(1)[:,None])  # nm
            Tijs[np.sum(T_thetas,axis=1) == 0, :] = 0

            # Edge normals subtracted from vertex normals
            Ni_diffs = np.sqrt(2. - 2.*np.sqrt(1.-((Nvi[None,:]*dvs_hat).sum(1))**2))  # unitless 
            Nj_diffs = np.sqrt(2. - 2.*np.sqrt(1.-((Nvjs*dvs_hat).sum(1))**2))  # unitless
            Nj_1_diffs = np.sqrt(2. - 2.*np.sqrt(1.-((Nvjs*dvs_1_hat).sum(1))**2))  # unitless

            # Compute the principal curvatures from the difference in normals (same as difference in tangents)
            kjs = 2.*Nj_diffs/dvs_norm  # 1/nm
            kjs_1 = 2.*Nj_1_diffs/dvs_1_norm  # 1/nm

            k = 2.*np.sign((Nvi[None,:]*dvs).sum(1))*Ni_diffs/dvs_norm  # 1/nm
            w = (1./dvs_norm)/r_sum  # unitless

            # Calculate areas
            Aj = self._faces['area'][self._halfedges['face'][neighbors[neighbor_mask]]]  # nm^2
            areas[iv] = np.sum(Aj)  # nm^2

            # Compute the change in bending energy along the edge (assumes no perpendicular contributions and thus no Gaussian curvature)
            dEj = Aj*w*self.kc*(2.0*kjs - self.c0)*(kjs_1 - kjs)/dN  # eV/nm

            Mvi = (w[None,:,None]*k[None,:,None]*Tijs.T[:,:,None]*Tijs[None,:,:]).sum(axis=1)  # nm

            l1, l2, v1, v2 = self._compute_curvature_tensor_eig(Mvi)

            # Eigenvectors
            m = np.vstack([v1, v2, Nvi]).T  # nm, nm, unitless

            # Principal curvatures
            k_1 = 3.*l1 - l2  # 1/nm
            k_2 = 3.*l2 - l1  # 1/nm

            # Mean and Gaussian curvatures
            H[iv] = 0.5*(k_1 + k_2)  # 1/nm
            K[iv] = k_1*k_2  # 1/nm^2

            # Now calculate the shift
            # We construct a quadratic in the space of T_1 vs. T_2
            t_1, t_2, _ = np.dot(vjs-vi,m).T  # nm^2
            A = np.array([t_1**2, t_2**2]).T  # nm^2
            
            # Update the equation y-intercept to displace the curve along
            # the normal direction
            b = np.dot(A,np.array([k_1,k_2])) - dN  # nm
            
            # Solve
            # Previously k_p, _, _, _ = np.linalg.lstsq(A, b)
            k_p = np.dot(np.dot(np.linalg.pinv(np.dot(A.T,A)),A.T),b)  # 1/nm

            # Finite differences of displaced curve and original curve
            dH[iv] = (0.5*(k_p[0] + k_p[1]) - H[iv])/dN  # 1/nm^2
            dK[iv] = ((k_p[0]-k_1)*k_2 + k_1*(k_p[1]-k_2))/dN  # 1/nm

            dE_neighbors[iv] = np.sum(dEj)  # eV/nm

        # Calculate Canham-Helfrich energy functional
        E = areas*(0.5*self.kc*(2.0*H - self.c0)**2 + self.kg*K)  # eV

        self._H = H  # 1/nm
        self._E = E  # eV
        self._K = K  # 1/nm^2
        
        self._pE = np.exp(-(1.0/self.kbt)*E)/(E.shape[0]*self._Q)  # unitless

        # Return probability of energy shift along direction of the normal
        return self._pE

    def point_attraction_prob(self, points, sigma, w=0.95, search_k=200, skip_prob=0.0):
        """
        Attractive force of membrane to points.

        Parameters
        ----------
            points : np.array
                3D point cloud to fit (nm).
            sigma : float
                Localization uncertainty of points (nm).
            w : float
                Weight (unitless)
            search_k : int
                Number of vertex point neighbors to consider
        """
        import scipy.spatial

        dirs = []

        charge_sigma = self._mean_edge_length/2.5  # nm
        charge_var = (2*charge_sigma**2)  # nm^2

        # Compute a KDTree on points
        tree = scipy.spatial.cKDTree(points)

        skip = np.random.rand(self._vertices.shape[0])

        for i in range(self._vertices.shape[0]):
            if self._vertices['halfedge'][i] != -1:
                # Monte carlo selection of vertices to update
                # Stochastically choose which vertices to adjust
                if skip[i] < skip_prob:
                    continue
                _, neighbors = tree.query(self._vertices['position'][i,:], search_k)
                try:
                    d = self._vertices['position'][i,:] - points[neighbors]  # nm
                except(IndexError):
                    print('whaaa?')
                    print(i, neighbors)
                dd = (d*d).sum(1)  # nm^2
                pt_weight_matrix = 1. - w*np.exp(-dd/charge_var)  # unitless
                pt_weights = np.prod(pt_weight_matrix)  # unitless
                r = np.sqrt(dd)/sigma[neighbors]  # unitless
                
                rf = -(1-r**2)*np.exp(-r**2/2) + (1-np.exp(-(r-1)**2/2))*(r/(r**3 + 1))  # unitless

                # Points at the vertex we're interested in are not de-weighted by the
                # pt_weight_matrix
                rf = rf*(pt_weights/pt_weight_matrix) # unitless
                
                # attraction = (-d*(rf/np.sqrt(dd))[:,None]).sum(0)  # unitless
                sign = -1.0*np.prod(np.sign((d*((self._vertices['normal'][i,:])[None,:])).sum(1)))
                attraction = sign*rf.sum(0)
            else:
                # attraction = np.array([0,0,0])
                attraction = 0.0
                        
            dirs.append(attraction)

        dirs = np.vstack(dirs).squeeze()
        dirs[self._vertices['halfedge'] == -1] = 0
        dirs = dirs/np.sum(np.abs(dirs))

        self._rf = np.abs(dirs)

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
        curvature = self.curvature_prob(dN=dN, skip_prob=self.skip_prob)
        attraction = self.point_attraction_prob(points, sigma, search_k=self.search_k, skip_prob=self.skip_prob)

        g = (attraction*curvature)[:,None]*self._vertices['normal']
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
                grad = self.c*self.curvature_prob(dN=dN)
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
            print('Unknown gradient descent method. Using {}.'.format(DEFAULT_DESCENT_METHOD))
            method = DEFAULT_DESCENT_METHOD

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
