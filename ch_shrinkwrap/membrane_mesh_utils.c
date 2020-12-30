/** @file membrane_mesh_utils.c
 *  @brief c helper functions for the _membrane_mesh pyx class
 * 
 * This file has expanded in scope to include some primary functions for 
 * _membrane_mesh.
 * 
 * @author zacsimile
 */

#include <stdio.h>

#include "Python.h"
#include <math.h>
#include "numpy/arrayobject.h"

#include "membrane_mesh_utils.h"

/** @brief Calculates the Euclidean norm of a vector.
 *
 *  Vector length is set by VECTORSIZE define.
 *
 *  @param pos A length VECTORSIZE vector (conventionally a position vector).
 *  @return PRECISION norm of the vector
 */
PRECISION norm(const PRECISION *pos)
{
    PRECISION n = 0;
    int i = 0;

    for (i = 0; i < VECTORSIZE; ++i)
        n += pos[i] * pos[i];
    return sqrt(n);
}

/** @brief Scalar division that returns 0 on div by 0
 *
 *  @param x PRECISION scalar
 *  @param y PRECISION scalar
 *  @return PRECISION division of x and y
 */
PRECISION safe_divide(PRECISION x, PRECISION y)
{
    if (y==0)
        return 0;
    return x/y;
}

/** @brief Sign of a scalar
 *
 *  @param x PRECISION scalar
 *  @return PRECISION sign of x
 */
PRECISION sign(PRECISION x)
{
    return (PRECISION)((0<x)-(x<0));
}


/** @brief Elementwise division of a vector by a scalar
 *
 *
 *  @param a PRECISION vector
 *  @param b PRECISION scalar
 *  @param c PRECISION vector b/a
 *  @param length int length of vector a
 *  @return Void
 */
void scalar_divide(const PRECISION *a, const PRECISION b, PRECISION *c, const int length)
{
    int k = 0;
    for (k=0; k < length; ++k)
        c[k] = a[k]/b;
}

/** @brief Elementwise multiplication of a scalar times a vector
 *
 *
 *  @param a PRECISION vector
 *  @param b PRECISION scalar
 *  @param c PRECISION vector b*a
 *  @param length int length of vector a
 *  @return Void
 */
void scalar_mult(const PRECISION *a, const PRECISION b, PRECISION *c, const int length)
{
    int k = 0;
    for (k=0; k < length; ++k)
        c[k] = a[k]*b;
}

/** @brief Construct outer product of vectors
 * 
 *  @param a PRECISION vector
 *  @param b PRECISION vector
 *  @param m PRECISION outer product
 *  @return Void
 */
void outer3(const PRECISION *a, const PRECISION *b, PRECISION *m)
{
    int i, j;
    for (i=0;i<3;++i)
    {
        for (j=0;j<3;++j)
        {
            m[i*3+j] = a[i]*b[j];
        }
    }
}

/** @brief Construct an orthogonal projection matrix
 *
 *  For a unit-normalized vector v, I-v*v.T is the projection
 *  matrix for the plane orthogonal to v
 * 
 *  @param v PRECISION unit-normalized vector
 *  @param m PRECISION orthogonal projection matrix
 *  @return Void
 */
void orthogonal_projection_matrix3(const PRECISION *v, PRECISION *m)
{
    PRECISION xy, xz, yz;

    xy = -v[0]*v[1];
    xz = -v[0]*v[2];
    yz = -v[1]*v[2];

    m[0] = 1-v[0]*v[0];
    m[1] = xy;
    m[2] = xz;
    m[3] = xy;
    m[4] = 1-v[1]*v[1];
    m[5] = yz;
    m[6] = xz;
    m[7] = yz;
    m[8] = 1-v[2]*v[2];
}

/** @brief Apply a 3x3 projection matrix to a 3-vector
 *
 * 
 *  @param p PRECISION orthogonal projection matrix
 *  @param v PRECISION vector
 *  @param r PRECISION projection of vector on plane defined by p
 *  @return Void
 */
void project3(const PRECISION *p, const PRECISION *v, PRECISION *r)
{
    r[0] = p[0]*v[0]+p[1]*v[1]+p[2]*v[2];
    r[1] = p[3]*v[0]+p[4]*v[1]+p[5]*v[2];
    r[2] = p[6]*v[0]+p[7]*v[1]+p[8]*v[2];
}

/** @brief Elementwise subtraction of two vectors
 *
 *  Subtracts vector b from vector a. Lengths of a and b must be equivalent.
 * 
 *  @param a PRECISION vector
 *  @param b PRECISION vector
 *  @param c PRECISION a-b
 *  @param length int length of vectors a and b
 *  @return Void
 */
void subtract(const PRECISION *a, const PRECISION *b, PRECISION *c, const int length)
{
    int i;
    for (i=0;i<length;++i)
        c[i] = a[i]-b[i];
}

/** @brief dot product of two vectors of equivalent length
 * 
 *  @param a PRECISION vector
 *  @param b PRECISION vector
 *  @param c PRECISION a <dot> b
 *  @param length int length of vectors a and b
 *  @return Void
 */
PRECISION dot(const PRECISION *a, const PRECISION *b, const int length)
{
    int i;
    PRECISION c;
    for (i=0;i<length;++i)
        c += a[i]*b[i];
    return c;
}

/** @brief Generate a pseudo-random number in [0,1)
 *
 *  borrowed from https://stackoverflow.com/questions/6218399/how-to-generate-a-random-number-between-0-and-1
 * 
 *  @return PRECISION pseudo-random number in [0,1)
 */
PRECISION r2()
{
    return (PRECISION)rand() / (PRECISION)((unsigned)RAND_MAX + 1);
}

static PyObject *calculate_pt_cnt_dist_2(PyObject *self, PyObject *args)
{
    PyArrayObject *points=0, *vertices=0, *pt_cnt_dist_2=0;
    points_t *p_points, *curr_point;
    vertex_t *p_vertices, *curr_vertex;
    int i, j, k, n_points, n_vertices, s_points, s_vertices;
    float tmp, tmp_diff, w, charge_sigma, charge_sigma_2;
    float *p_pt_cnt_dist_2;

    if (!PyArg_ParseTuple(args, "OOOff", &points, &vertices, &pt_cnt_dist_2, &w, &charge_sigma)) return NULL;
    if (!PyArray_Check(points) || !PyArray_ISCONTIGUOUS(points))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the points data.");
        return NULL;
    }
    if (!PyArray_Check(vertices) || !PyArray_ISCONTIGUOUS(vertices)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the vertex data.");
        return NULL;
    }
    if (!PyArray_Check(pt_cnt_dist_2) || !PyArray_ISCONTIGUOUS(pt_cnt_dist_2)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the pt_cnt_dist_2 data.");
        return NULL;
    }

    p_points = (points_t*)PyArray_GETPTR1(points, 0);
    p_vertices = (vertex_t*)PyArray_GETPTR1(vertices, 0);
    p_pt_cnt_dist_2 = (float*)PyArray_GETPTR2(pt_cnt_dist_2, 0, 0);

    n_points = PyArray_SHAPE(pt_cnt_dist_2)[0];
    n_vertices = PyArray_SHAPE(pt_cnt_dist_2)[1];
    // printf("%d   %d\n", n_points, n_vertices);
    s_points = PyArray_STRIDE(pt_cnt_dist_2, 0);
    s_vertices = PyArray_STRIDE(pt_cnt_dist_2, 1);

    // printf("n_points: %d    n_vertices: %d    s_points: %d    s_vertices: %d\n", n_points, n_vertices, s_points, s_vertices);

    charge_sigma_2 = 2*charge_sigma*charge_sigma;

    for (j = 0; j < n_vertices; ++j)
    {
        curr_vertex = &(p_vertices[j]);
        for (i = 0; i < n_points; ++i)
        {
            // printf("i: %d     j: %d\n", i, j);
            curr_point = &(p_points[i]);
            tmp = 0;
            for (k = 0; k < VECTORSIZE; ++k)
                tmp_diff = (curr_point->position[k]) - (curr_vertex->position[k]);
                tmp += 1 - w*exp(-tmp_diff*tmp_diff/charge_sigma_2);
            // printf("tmp: %f\n", tmp);
            p_pt_cnt_dist_2[i*n_vertices+j] = tmp;
            //*(p_pt_cnt_dist_2 + (i*s_vertices + j*s_points)) = tmp;
        }
    }

    Py_INCREF(Py_None);
    return Py_None;

}

/** @brief calculate point weight matrix and point weights for point attraction force
 *
 * 
 *  @param points points_t (n_points x 3) matrix of point cloud positions
 *  @param vertices_ void vertex_d/vertex_t set of mesh vertices
 *  @param pt_weight_matrix PRECISION (n_points x n_vertices) weight matrix
 *  @param pt_weights PRECISION n_points weight array
 *  @param w float additional weight prefactor
 *  @param charge_sigma PRECISION distance between a point and a vertex at which charge shielding occurs
 *  @param n_points int number of points in the point cloud we are wrapping
 *  @param n_vertices int number of vertices in the mesh
 *  @return Void
 */
static void calc_pt_weight_matrix(points_t *points, 
                                  void *vertices_, 
                                  PRECISION *pt_weight_matrix, 
                                  PRECISION *pt_weights, 
                                  PRECISION w, 
                                  PRECISION charge_sigma, 
                                  int n_points, 
                                  int n_vertices)
{
    int i, j, k;
    PRECISION charge_sigma_2, tmp, tmp_diff;
    points_t *curr_point;
    vertex_t *curr_vertex;
    vertex_t *vertices = (vertex_t*) vertices_;

    charge_sigma_2 = 2*charge_sigma*charge_sigma;

    // pre-fill pt_weights for *= operations
    for (i=0;i<n_points;++i)
        pt_weights[i] = 1;

    for (j = 0; j < n_vertices; ++j)
    {
        curr_vertex = &(vertices[j]);
        for (i = 0; i < n_points; ++i)
        {
            // printf("i: %d     j: %d\n", i, j);
            curr_point = &(points[i]);
            tmp = 0;
            for (k = 0; k < VECTORSIZE; ++k)
                tmp_diff = (curr_point->position[k]) - (curr_vertex->position[k]);
                tmp += 1 - w*exp(-tmp_diff*tmp_diff/charge_sigma_2);
            // printf("tmp: %f\n", tmp);
            pt_weight_matrix[i*n_vertices+j] = tmp;
            //*(p_pt_cnt_dist_2 + (i*s_vertices + j*s_points)) = tmp;
            pt_weights[i] *= tmp;
        }
    }

}


/** @brief calculate point attraction force gradient
 *
 * 
 *  @param attraction points_t (n_points x 3) matrix of the attraction force gradient
 *  @param points points_t (n_points x 3) matrix of point cloud positions
 *  @param sigma PRECISION vector of point localization precisions (1D, e.g. error_x)
 *  @param vertices vertex_t set of mesh vertices
 *  @param w float additional weight prefactor
 *  @param charge_sigma PRECISION distance between a point and a vertex at which charge shielding occurs
 *  @param n_points int number of points in the point cloud we are wrapping
 *  @param n_vertices int number of vertices in the mesh
 *  @return Void
 */
static void c_point_attraction_grad(points_t *attraction, 
                             points_t *points, 
                             PRECISION *sigma, 
                             vertex_t *vertices, 
                             PRECISION w, 
                             PRECISION charge_sigma, 
                             int n_points, 
                             int n_vertices)
{
    int i, j, k;
    int32_t curr_idx;
    PRECISION *pt_weight_matrix;
    PRECISION *pt_weights;
    vertex_t *curr_vertex;
    points_t *curr_point, *curr_attraction;
    PRECISION d[VECTORSIZE];
    PRECISION dd, r, r2, r12, rf;

    pt_weight_matrix = (PRECISION *)malloc(sizeof(PRECISION)*n_points*n_vertices);  // n_points x n_vertices
    pt_weights = (PRECISION *)malloc(sizeof(PRECISION)*n_points); // n_points
    calc_pt_weight_matrix(points, vertices, pt_weight_matrix, pt_weights, w, charge_sigma, n_points, n_vertices);

    for (i=0;i<n_vertices;++i)
    {
        curr_vertex = &(vertices[i]);
        curr_idx = curr_vertex->halfedge;

        curr_attraction = &(attraction[i]);

        if (curr_idx == -1) 
        {
            for (k=0;k<VECTORSIZE;++k)
                (curr_attraction->position[k]) = 0;
            continue;
        }

        for (j=0;j<n_points;++j)
        {
            curr_point = &(points[j]);
            for (k=0;k<VECTORSIZE;++k)
                d[k] = curr_vertex->position[k] - curr_point->position[k];
            dd = norm(d);

            r = dd/sigma[j];
            r2 = r*r; r12 = (r-1)*(r-1);
            rf = -(1-r2)*exp(-r2/2) + (1-exp(-r12/2))*(r/(r2*r + 1));
            rf *= (pt_weights[j]/pt_weight_matrix[j*n_vertices+i]);
            for (k=0;k<VECTORSIZE;++k)
                curr_attraction->position[k] += -d[k]*(rf/dd);
        }
    }
    free(pt_weight_matrix);
    free(pt_weights);
}

/** @brief find eigenvalues/vectors of 3x3 curvature tensor using closed-form solution
 *
 *  @param Mvi PRECISION (3x3) matrix describing local curvature
 *  @param l1 PRECISION first non-zero eigenvalue of Mvi
 *  @param l2 PRECISION second non-zero eigenvalue of Mvi
 *  @param v1 PRECISION eigenvector of Mvi associated with l1
 *  @param v2 PRECISION eigenvector of Mvi associated with l2
 *  @return Void
 */
static void compute_curvature_tensor_eig(PRECISION *Mvi, PRECISION *l1, PRECISION *l2, PRECISION *v1, PRECISION *v2) 
{
    PRECISION m00, m01, m02, m11, m12, m22, p, q, r, z1n, z1d, z1, y1n, y1d, y1, z2n, z2d, z2, y2n, y2d, y2;
    PRECISION v2t[3];
    PRECISION v1t[3];

    m00 = Mvi[0];
    m01 = Mvi[1];
    m02 = Mvi[2];
    m11 = Mvi[4];
    m12 = Mvi[5];
    m22 = Mvi[8];

    // Here we use the fact that Mvi is symnmetric and we know
    // one of the eigenvalues must be 0
    p = -m00*m11 - m00*m22 + m01*m01 + m02*m02 - m11*m22 + m12*m12;
    q = m00 + m11 + m22;
    r = sqrt(4*p + q*q);

    // Eigenvalues
    *l1 = 0.5*(q-r);
    *l2 = 0.5*(q+r);

    // Now calculate the eigenvectors, assuming x = 1
    z1n = ((m00 - (*l1))*(m11 - (*l1)) - (m01*m01));
    z1d = (m01*m12 - m02*(m11 - (*l1)));
    z1 = safe_divide(z1n, z1d);
    y1n = (m12*z1 + m01);
    y1d = (m11 - (*l1));
    y1 = safe_divide(y1n, y1d);
    
    v1t[0] = 1; v1t[1] = y1; v1t[2] = z1;
    scalar_divide(v1t,norm(v1t),v1,VECTORSIZE);
    
    z2n = ((m00 - (*l2))*(m11 - (*l2)) - (m01*m01));
    z2d = (m01*m12 - m02*(m11 - (*l2)));
    z2 = safe_divide(z2n, z2d);
    y2n = (m12*z2 + m01);
    y2d = (m11 - (*l2));
    y2 = safe_divide(y2n, y2d);
    
    v2t[0] = 1; v2t[1] = y2; v2t[2] = z2;
    scalar_divide(v2t,norm(v2t),v2,VECTORSIZE);
}

/** @brief c implementation to calculate gradient of canham-helfrich energy function at mesh vertices
 * 
 *  Where possible, I have tried to match the code (order, variable names) to the original python code.
 *
 *  @param vertices_ void vertex_d/vertex_t set of mesh vertices
 *  @param faces_ void face_d/face_t set of mesh faces
 *  @param halfedges halfedges_t set of mesh halfedges
 *  @param dN PRECISION displacment of the normal vector for finite difference estimation of curvature
 *  @param skip_prob PRECISION optional probability in [0,1] for Monte-Carlo subsampling of vertices during update step
 *  @param n_vertices int number of vertices in the mesh
 *  @param H PRECISION n_vertices vector of mean curvatures 
 *  @param K PRECISION n_vertices vector of Gaussian curvatures 
 *  @param dH PRECISION n_vertices vector of mean curvature gradient
 *  @param dK PRECISION n_vertices vector of Gaussian curvature gradient
 *  @param E PRECISION n_vertices vector of bending energy, as defined by Canham-Helfrich functional, stored in each vertex
 *  @param pE PRECISION n_vertices vector of energy likelihood, defined by a Boltzmann distribution on the Canham-Helfrich
 *  @param dE_neighbors n_vertices vector of how a shift at this vertex changes the bending energy in 1-ring neighbor vertices
 *  @param kc PRECISION bending stiffness coefficient (eV)
 *  @param kg PRECISION bending stiffness coefficient (eV)
 *  @param c0 PRECISION spontaneous curvature (1/nm)
 *  @param dEdN n_vertices vector of energy gradient
 *  @return Void
 */
static void c_curvature_grad(void *vertices_, 
                             void *faces_,
                             halfedge_t *halfedges,
                             PRECISION dN,
                             PRECISION skip_prob,
                             int n_vertices,
                             PRECISION *H,
                             PRECISION *K,
                             PRECISION *dH,
                             PRECISION *dK,
                             PRECISION *E,
                             PRECISION *pE,
                             PRECISION *dE_neighbors,
                             PRECISION kc,
                             PRECISION kg,
                             PRECISION c0,
                             points_t *dEdN)
{
    int i, j, q, neighbor, n_neighbors;
    PRECISION l1, l2, r_sum, dv_norm, dv_1_norm, T_theta_norm, Ni_diff, Nj_diff, Nj_1_diff;
    PRECISION kj, kj_1, k, Aj, areas, w, k_1, k_2;
    PRECISION v1[VECTORSIZE], v2[VECTORSIZE], Mvi[VECTORSIZE*VECTORSIZE];
    PRECISION Mvi_temp[VECTORSIZE*VECTORSIZE], Mvi_temp2[VECTORSIZE*VECTORSIZE];
    PRECISION m[VECTORSIZE*VECTORSIZE];
    PRECISION p[VECTORSIZE*VECTORSIZE], dv[VECTORSIZE], ndv[VECTORSIZE];
    PRECISION dv_hat[VECTORSIZE], dv_1[VECTORSIZE], dv_1_hat[VECTORSIZE];
    PRECISION NvidN[VECTORSIZE], T_theta[VECTORSIZE], Tij[VECTORSIZE];
    PRECISION A[NEIGHBORSIZE], b[NEIGHBORSIZE];
    PRECISION *vi, *vj, *Nvi, *Nvj;
    vertex_t *curr_vertex, *neighbor_vertex;
    halfedge_t *curr_neighbor;
    vertex_t *vertices = (vertex_t*) vertices_;
    face_t *faces = (face_t*) faces_;

    for (i=0;i<n_vertices;++i)
    {
        curr_vertex = &(vertices[i]);
        if ((curr_vertex->halfedge) == -1)
            continue;

        // Monte carlo selection of vertices to update
        // Stochastically choose which vertices to adjust
        if ((skip_prob>0)&&(r2()<skip_prob))
            continue;
        
        // Vertex and its normal
        vi = curr_vertex->position; // nm
        Nvi = curr_vertex->normal;  // unitless

        // projection matrix
        scalar_mult(Nvi, dN, NvidN, VECTORSIZE);  // unitless
        orthogonal_projection_matrix3(Nvi, p);  // unitless

        // Need a three-pass over the neighbors: 1. get the radial weights
        r_sum = 0; // 1/nm
        j = 0;
        neighbor = (curr_vertex->neighbors)[j];
        while(neighbor!=-1)
        {
            curr_neighbor = &(halfedges[neighbor]);
            neighbor_vertex = &(vertices[curr_neighbor->vertex]);

            vj = neighbor_vertex->position;  // nm
            subtract(vi,vj,dv,VECTORSIZE); // nm
            dv_norm = norm(dv);  // nm
            // radial weighting
            r_sum += 1.0/dv_norm;  // 1/nm

            ++j;
            neighbor = (curr_vertex->neighbors)[j];
        }
        n_neighbors = j;  // record the number of neighbors for later passes

        // zero out Mvi
        for (j=0;j<VECTORSIZE*VECTORSIZE;++j)
            Mvi[j] = 0;

        // 2. Compute Mvi
        areas = 0;  // nm^2
        dE_neighbors[i] = 0;  // eV/nm
        for(j=0;j<n_neighbors;++j)
        {
            neighbor = (curr_vertex->neighbors)[j];
            curr_neighbor = &(halfedges[neighbor]);
            neighbor_vertex = &(vertices[curr_neighbor->vertex]);
            vj = neighbor_vertex->position;  // nm
            subtract(vi,vj,dv,VECTORSIZE); // nm
            subtract(dv,NvidN,dv_1,VECTORSIZE);  // nm

            dv_norm = norm(dv);  // nm
            dv_1_norm = norm(dv_1);  // nm

            // normalized vectors
            scalar_divide(dv,dv_norm,dv_hat,VECTORSIZE);  // unitless
            scalar_divide(dv_1,dv_1_norm,dv_1_hat,VECTORSIZE);  // unitless

            // tangents
            scalar_mult(dv,-1.0,ndv,VECTORSIZE); // nm
            project3(p, dv, T_theta); // nm^2
            T_theta_norm = norm(T_theta); // nm^2
            if (T_theta_norm>0)
            {
                scalar_divide(T_theta,T_theta_norm,Tij,VECTORSIZE); // unitless
            } 
            else
            {
                for (q=0;q<VECTORSIZE;++q)
                    Tij[q] = 0;
            }

            // edge normals subtracted from vertex normals
            Ni_diff = sqrt(2.0-2.0*sqrt(1.0-pow(dot(Nvi,dv_hat,VECTORSIZE),2.0)));  // 1/nm
            Nvj = neighbor_vertex->normal;  // unitless
            Nj_diff = sqrt(2.0-2.0*sqrt(1.0-pow(dot(Nvj,dv_hat,VECTORSIZE),2.0)));  // 1/nm
            Nj_1_diff = sqrt(2.0-2.0*sqrt(1.0-pow(dot(Nvj,dv_1_hat,VECTORSIZE),2.0)));  // 1/nm

            // Compute the principal curvatures from the difference in normals (same as difference in tangents)
            kj = 2.0*Nj_diff/dv_norm;  // 1/nm
            kj_1 = 2.0*Nj_1_diff/dv_1_norm; // 1/nm

            // weights/areas
            w = (1.0/dv_norm)/r_sum;
            k = 2.0*sign(dot(Nvi,dv,VECTORSIZE))*Ni_diff/dv_norm;  // 1/nm
            Aj = faces[curr_neighbor->face].area;  // nm^2
            areas += Aj;  // nm^2
            dE_neighbors[i] += Aj*w*kc*(2.0*kj-c0)*(kj_1-kj)/dN;  // eV/nm

            // Construct Mvi
            outer3(Tij,Tij,Mvi_temp);
            scalar_mult(Mvi_temp,w*k,Mvi_temp2,VECTORSIZE*VECTORSIZE);
            for (q=0;q<VECTORSIZE*VECTORSIZE;++q)
                Mvi[q] += Mvi_temp[q];
        }

        // Interlude: calculate curvature tensor
        compute_curvature_tensor_eig(Mvi, &l1, &l2, v1, v2);

        // principal curvatures
        k_1 = 3.0*l1 - l2;
        k_2 = 3.0*l2 - l1;

        // mean and gaussian curvatures
        H[i] = 0.5*(k_1+k_2);
        K[i] = k_1*k_2;

        // create little m (eigenvector matrix)
        m[0] = v1[0]; m[3] = v1[1]; m[6] = v1[2];
        m[1] = v2[1]; m[4] = v2[1]; m[7] = v2[2];
        m[2] = Nvi[0]; m[5] = Nvi[1]; m[8] = Nvi[2];  // TODO: we don't need these assignments, skip?

        // 3. Compute shift
        for (j=0;j<n_neighbors;++j)
        {
            neighbor = (curr_vertex->neighbors)[j];
            curr_neighbor = &(halfedges[neighbor]);
            neighbor_vertex = &(vertices[curr_neighbor->vertex]);
            vj = neighbor_vertex->position;  // nm

            subtract(vj,vi,dv,VECTORSIZE); // nm

            // construct a quadratic in the space of T_1 vs. T_2
            A[2*j+0] = dv[0]*m[0]+dv[1]*m[3]+dv[2]*m[6];
            A[2*j+1] = dv[0]*m[1]+dv[1]*m[4]+dv[2]*m[7];

            // Update the equation y-intercept to displace athe curve along the normal direction
            b[j] = A[2*j+0]*k_1+A[2*j+1]*k_2;
        }

    }
}


static PyMethodDef membrane_mesh_utils_methods[] = {
    {"calculate_pt_cnt_dist_2", calculate_pt_cnt_dist_2, METH_VARARGS},
    {NULL, NULL, 0}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "membrane_mesh_utils",     /* m_name */
        "C implementations of membrane_mesh operations for speed improvement",  /* m_doc */
        -1,                  /* m_size */
        membrane_mesh_utils_methods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };

PyMODINIT_FUNC PyInit_membrane_mesh_utils(void)
{
	PyObject *m;
    m = PyModule_Create(&moduledef);
    import_array()
    return m;
}
#else
PyMODINIT_FUNC initmembrane_mesh_utils(void)
{
    PyObject *m;

    m = Py_InitModule("membrane_mesh_utils", membrane_mesh_utils_methods);
    import_array()

}
#endif