/** @file membrane_mesh_utils.c
 *  @brief c helper functions for the _membrane_mesh pyx class
 * 
 * This file has expanded in scope to include some primary functions for 
 * _membrane_mesh.
 * 
 * @author zacsimile
 */

#include <stdio.h>
#include <stdbool.h>
#include <float.h>

#include "Python.h"
#include <math.h>
#include "numpy/arrayobject.h"

#include "membrane_mesh_utils.h"

#define ABS(x) (((x) < 0) ? -x : x)
#define SQUARE(a) ((a)*(a))
#define SIGN(x) (((x) < 0) ? -1 : 1)

#define EPSILON 1e-15

/** @brief Calculates the Euclidean norm of a vector.
 *
 *  @param pos A length 3 double vector (conventionally a position vector).
 *  @return double norm of the vector
 */
double norm3(const double *pos)
{
    int i;
    double n = 0.0;

    for (i = 0; i<3; ++i)
        n += (pos[i]) * (pos[i]);
    return sqrt(n);
}

/** @brief Calculates the Euclidean norm of a vector.
 *
 *  @param pos A length 3 float vector (conventionally a position vector).
 *  @return double norm of the vector
 */
float fnorm3f(const float *pos)
{
    int i;
    float n = 0.0;

    for (i = 0; i<3; ++i)
        n += (pos[i]) * (pos[i]);
    return sqrt(n);
}

/** @brief Scalar division that returns 0 on div by 0
 *
 *  @param x double scalar
 *  @param y double scalar
 *  @return double division of x and y
 */
double safe_divide(double x, double y)
{
    if (ABS(y)<EPSILON)
        return 0.0;
    return (x)/(y);
}

/** @brief Scalar division that returns 0 on div by 0
 *
 *  @param x float scalar
 *  @param y float scalar
 *  @return float division of x and y
 */
float ffsafe_dividef(float x, float y)
{
    if (ABS(y)<EPSILON)
        return 0.0;
    return (x)/(y);
}

/** @brief Elementwise division of a vector by a scalar
 *
 *
 *  @param a double vector
 *  @param b double scalar
 *  @param c double vector b/a
 *  @return Void
 */
void scalar_divide3(const double *a, const double b, double *c)
{
    int k = 0;
    for (k=0; k<3; ++k)
        c[k] = (a[k])/(b);
}

/** @brief Elementwise division of a vector by a scalar
 *
 *
 *  @param a float vector
 *  @param b float scalar
 *  @param c float vector b/a
 *  @return Void
 */
void ffscalar_divide3f(const float *a, const float b, float *c)
{
    int k = 0;
    for (k=0; k<3; ++k)
        c[k] = (a[k])/(b);
}

/** @brief Elementwise multiplication of a scalar times a vector
 *
 *
 *  @param a double vector
 *  @param b double scalar
 *  @param c double vector b*a
 *  @param length int length of vector a
 *  @return Void
 */
void scalar_mult(const double *a, const double b, double *c, const int length)
{
    int k = 0;
    for (k=0; k < length; ++k)
        c[k] = (a[k])*(b);
}

/** @brief Elementwise multiplication of a scalar times a vector
 *
 *
 *  @param a float vector
 *  @param b float scalar
 *  @param c double vector b*a
 *  @param length int length of vector a
 *  @return Void
 */
void ffscalar_multd(const float *a, const float b, double *c, const int length)
{
    int k = 0;
    for (k=0; k < length; ++k)
        c[k] = ((double)(a[k]))*((double)(b));
}

/** @brief Construct outer product of vectors
 * 
 *  @param a double vector
 *  @param b double vector
 *  @param m double outer product
 *  @return Void
 */
void outer3(const double *a, const double *b, double *m)
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

/** @brief Multiply two matrices
 * 
 *  @param a double vector (m x n)
 *  @param b double vector (n x p)
 *  @param c double outer product
 *  @param m int first dimension of a
 *  @param n int second dimension of a/first of b
 *  @param p int second dimension of b
 *  @return Void
 */
void matmul(const double *a, const double *b, double *c, int m, int n, int p)
{
    int i, j, k;

    for (i=0;i<m;++i)
    {
        for (j=0;j<p;++j)
        {
            c[i*p+j] = 0.0;  // zero it out first
            for (k=0;k<n;++k)
                c[i*p+j] += a[i*n+k]*b[k*p+j];
        }
    }
}

/** @brief Transpose a matrix
 * 
 *  @param a double matrix (m x n)
 *  @param at double matrix (n x m)
 *  @param m int first dimension of a
 *  @param n int second dimension of a
 *  @return Void
 */
void transpose(const double *a, double *at, int m, int n)
{
    int i, j;
    for (i=0;i<m;++i) {
        for (j=0;j<n;++j)
            at[j*m+i] = a[i*n+j];
    }
}

/** @brief Construct an orthogonal projection matrix
 *
 *  For a unit-normalized vector v, I-v*v.T is the projection
 *  matrix for the plane orthogonal to v
 * 
 *  @param v PRECISION unit-normalized vector
 *  @param m double orthogonal projection matrix
 *  @return Void
 */
void orthogonal_projection_matrix3(const PRECISION *v, double *m)
{
    PRECISION xy, xz, yz;
    double v0, v1, v2;

    v0 = (double)(v[0]);
    v1 = (double)(v[1]);
    v2 = (double)(v[2]);

    xy = -1.0*v0*v1;
    xz = -1.0*v0*v2;
    yz = -1.0*v1*v2;

    m[0] = 1.0-v0*v0;
    m[1] = xy;
    m[2] = xz;
    m[3] = xy;
    m[4] = 1.0-v1*v1;
    m[5] = yz;
    m[6] = xz;
    m[7] = yz;
    m[8] = 1.0-v2*v2;
}

/** @brief Apply a 3x3 projection matrix to a 3-vector
 *
 * 
 *  @param p double orthogonal projection matrix
 *  @param v double vector
 *  @param r double projection of vector on plane defined by p
 *  @return Void
 */
void project3(const double *p, const double *v, double *r)
{
    r[0] = p[0]*v[0]+p[1]*v[1]+p[2]*v[2];
    r[1] = p[3]*v[0]+p[4]*v[1]+p[5]*v[2];
    r[2] = p[6]*v[0]+p[7]*v[1]+p[8]*v[2];
}

/** @brief Elementwise subtraction of two vectors
 *
 *  Subtracts vector b from vector a. Lengths of a and b must be equivalent.
 * 
 *  @param a double vector
 *  @param b double vector
 *  @param c double a-b
 *  @return Void
 */
void subtract3(const double *a, const double *b, double *c)
{
    int i;
    for (i=0;i<3;++i)
        c[i] = a[i]-b[i];
}

/** @brief Elementwise subtraction of two vectors
 * 
 *  @param a float vector
 *  @param b double vector
 *  @param c double a-b
 *  @return Void
 */
void fdsubtract3d(const float *a, const double *b, double *c)
{
    int i;
    for (i=0;i<3;++i)
        c[i] = ((double)(a[i]))-b[i];
}

/** @brief Elementwise subtraction of two vectors
 * 
 *  @param a float vector
 *  @param b float vector
 *  @param c double a-b
 *  @return Void
 */
void ffsubtract3d(const float *a, const float *b, double *c)
{
    int i;
    for (i=0;i<3;++i)
        c[i] = ((double)(a[i]))-((double)(b[i]));
}

/** @brief Elementwise subtraction of two vectors
 * 
 *  @param a float vector
 *  @param b float vector
 *  @param c double a-b
 *  @return Void
 */
void ffsubtract3f(const float *a, const float *b, float *c)
{
    int i;
    for (i=0;i<3;++i)
        c[i] = (a[i])-(b[i]);
}

/** @brief dot product of two vectors of equivalent length
 * 
 *  @param a double vector
 *  @param b double vector
 *  @return double a <dot> b
 */
double dot3(const double *a, const double *b)
{
    int i;
    double c = 0.0;
    for (i=0;i<3;++i)
        c += a[i]*b[i];
    return c;
}

/** @brief dot product of two vectors of equivalent length
 * 
 *  @param a float vector
 *  @param b double vector
 *  @return double a <dot> b
 */
double fddot3d(const float *a, const double *b)
{
    int i;
    double c = 0.0;
    for (i=0;i<3;++i)
        c += ((double)(a[i]))*b[i];
    return c;
}

/** @brief cross product of two vectors of equivalent length
 * 
 *  @param a double vector
 *  @param b double vector
 *  @param c double a <cross> b
 *  @return Void
 */
void cross3(const double *a, const double *b, double *n)
{
    n[0] = a[1]*b[2] - a[2]*b[1];
    n[1] = a[2]*b[0] - a[0]*b[2];
    n[2] = a[0]*b[1] - a[1]*b[0];
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
            dd = fnorm3f(d);

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
 *  @param Mvi double (3x3) matrix describing local curvature
 *  @param l1 double first non-zero eigenvalue of Mvi
 *  @param l2 double second non-zero eigenvalue of Mvi
 *  @param v1 double eigenvector of Mvi associated with l1
 *  @param v2 double eigenvector of Mvi associated with l2
 *  @return Void
 */
static void compute_curvature_tensor_eig(double *Mvi, double *l1, double *l2, double *v1, double *v2) 
{
    double m00, m01, m02, m11, m12, m22, p, q, r, z1n, z1d, z1, y1n, y1d, y1, z2n, z2d, z2, y2n, y2d, y2;
    double v2t[3];
    double v1t[3];

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
    scalar_divide3(v1t,norm3(v1t),v1);
    
    z2n = ((m00 - (*l2))*(m11 - (*l2)) - (m01*m01));
    z2d = (m01*m12 - m02*(m11 - (*l2)));
    z2 = safe_divide(z2n, z2d);
    y2n = (m12*z2 + m01);
    y2d = (m11 - (*l2));
    y2 = safe_divide(y2n, y2d);
    
    v2t[0] = 1; v2t[1] = y2; v2t[2] = z2;
    scalar_divide3(v2t,norm3(v2t),v2);
    
}

/** @brief find eigenvalues/vectors of 3x3 curvature tensor using closed-form solution
 *
 *  @param Mvi float (3x3) matrix describing local curvature
 *  @param l1 float first non-zero eigenvalue of Mvi
 *  @param l2 float second non-zero eigenvalue of Mvi
 *  @param v1 float eigenvector of Mvi associated with l1
 *  @param v2 float eigenvector of Mvi associated with l2
 *  @return Void
 */
static void fcompute_curvature_tensor_eig(float *Mvi, float *l1, float *l2, float *v1, float *v2) 
{
    float m00, m01, m02, m11, m12, m22, p, q, r, z1n, z1d, z1, y1n, y1d, y1, z2n, z2d, z2, y2n, y2d, y2;
    float v2t[3];
    float v1t[3];

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
    z1 = ffsafe_dividef(z1n, z1d);
    y1n = (m12*z1 + m01);
    y1d = (m11 - (*l1));
    y1 = ffsafe_dividef(y1n, y1d);
    
    v1t[0] = 1; v1t[1] = y1; v1t[2] = z1;
    ffscalar_divide3f(v1t,fnorm3f(v1t),v1);
    
    z2n = ((m00 - (*l2))*(m11 - (*l2)) - (m01*m01));
    z2d = (m01*m12 - m02*(m11 - (*l2)));
    z2 = ffsafe_dividef(z2n, z2d);
    y2n = (m12*z2 + m01);
    y2d = (m11 - (*l2));
    y2 = ffsafe_dividef(y2n, y2d);
    
    v2t[0] = 1; v2t[1] = y2; v2t[2] = z2;
    ffscalar_divide3f(v2t,fnorm3f(v2t),v2);
    
}

/** @brief Compute pseudoinverse of 2 x 2 matrix
 * 
 *  Closed form of Moore-Penrose pseudoinverse for a 2x2 matrix.
 *
 *  @param A double matrix
 *  @param Ainv double inverted matrix
 *  @return Void
 */
void moore_penrose_2x2(const double *A, double *Ainv)
{
    double a,b,c,d,a2,b2,c2,d2,a2b2,c2d2,a2b2nc2nd2,tacbd;
    double theta, phi, ctheta, cphi, stheta, sphi;
    double ctcp, ctsp, stcp, stsp, sign0, sign1, ss, sd, sssd;
    double sig0, sig1, thresh, siginv0, siginv1, s0s0, s1s1;
    a = A[0]; b = A[1]; c = A[2]; d = A[3];

    a2 = a*a; b2 = b*b; c2 = c*c; d2 = d*d;
    
    a2b2 = a2+b2; c2d2 = c2+d2;
    a2b2nc2nd2 = a2b2-c2d2;
    tacbd = 2*(a*c+b*d);
    
    theta = 0.5*atan2(2*(a*b+c*d),a2+c2-b2-d2);
    phi = 0.5*atan2(tacbd, a2b2nc2nd2);
    
    ctheta = cos(theta); cphi = cos(phi);
    stheta = sin(theta); sphi = sin(phi);
    
    ctcp = ctheta*cphi; ctsp = ctheta*sphi;
    stcp = stheta*cphi; stsp = stheta*sphi;
    
    sign0 = SIGN(ctcp*a+ctsp*c+stcp*b+stsp*d);
    sign1 = SIGN(stsp*a-stcp*c-ctsp*b+ctcp*d);
    
    ss = a2b2+c2d2;
    sd = sqrt(SQUARE(a2b2nc2nd2)+SQUARE(tacbd));
    
    sig0 = sqrt((ss+sd)/2.0); 
    sssd = ss-sd;

    if (sssd > 0)
        sig1 = sqrt(sssd/2.0);
    else
        sig1 = 0.0;
    
    thresh = (1e-8)*0.5*sqrt(5.0)*sig0;

    siginv0 = (sig0 < thresh) ? 0.0 : (1.0/sig0);
    siginv1 = (sig1 < thresh) ? 0.0 : (1.0/sig1);
    
    s0s0 = sign0*siginv0; s1s1 = sign1*siginv1;
    
    Ainv[0] = (ctcp*s0s0+stsp*s1s1);
    Ainv[1] = (ctsp*s0s0-stcp*s1s1);
    Ainv[2] = (stcp*s0s0-ctsp*s1s1);
    Ainv[3] = (stsp*s0s0+ctcp*s1s1);
    
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
 *  @param dE_neighbors PRECISION n_vertices vector of how a shift at this vertex changes the bending energy in 1-ring neighbor vertices
 *  @param kc PRECISION bending stiffness coefficient (eV)
 *  @param kg PRECISION bending stiffness coefficient (eV)
 *  @param c0 PRECISION spontaneous curvature (1/nm)
 *  @param dEdN PRECISION n_vertices vector of energy gradient
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
    int i, j, jj, neighbor, n_neighbors;
    double l1, l2, r_sum, dv_norm, dv_1_norm, T_theta_norm, Ni_diff, Nj_diff, Nj_1_diff;
    double kj, kj_1, k, Aj, areas, dareas, w, k_1, k_2;
    double dEdN_H, dEdN_K, dEdN_sum;
    double Nvidv_hat, Nvjdv_hat, Nvjdv_1_hat;
    double v1[VECTORSIZE], v2[VECTORSIZE], Mvi[VECTORSIZE*VECTORSIZE];
    double Mvi_temp[VECTORSIZE*VECTORSIZE], Mvi_temp2[VECTORSIZE*VECTORSIZE];
    double m[VECTORSIZE*VECTORSIZE];
    double p[VECTORSIZE*VECTORSIZE], dv[VECTORSIZE], dvn[VECTORSIZE], dv_1dvn[VECTORSIZE], ndv[VECTORSIZE];
    double dv_hat[VECTORSIZE], dv_1[VECTORSIZE], dv_1_hat[VECTORSIZE];
    double NvidN[VECTORSIZE], viNvidN[VECTORSIZE], T_theta[VECTORSIZE], Tij[VECTORSIZE];
    double A[2*NEIGHBORSIZE], At[2*NEIGHBORSIZE], AtA[4], AtAinv[4], AtAinvAt[2*NEIGHBORSIZE];
    double b[NEIGHBORSIZE], k_p[2];
    PRECISION dEdNs;
    PRECISION *vi, *vj, *vn, *Nvi, *Nvj;
    vertex_t *curr_vertex, *neighbor_vertex, *next_neighbor_vertex;
    halfedge_t *curr_neighbor, *next_neighbor;
    vertex_t *vertices = (vertex_t*) vertices_;
    face_t *faces = (face_t*) faces_;

    for (i=0;i<n_vertices;++i)
    {
        curr_vertex = &(vertices[i]);
        // Skip unused vertices || stochastically choose which vertices to adjust
        if ( ((curr_vertex->halfedge) == -1) || ((skip_prob>0)&&(r2()<skip_prob)) ) {
            H[i] = 0.0;
            K[i] = 0.0;
            dH[i] = 0.0;
            dK[i] = 0.0;
            dE_neighbors[i] = 0.0;
            E[i] = 0.0;
            pE[i] = 0.0;
            for (jj=0;jj<VECTORSIZE;++jj)
                (dEdN[i]).position[jj] = 0.0;
            continue;
        }

        // Vertex and its normal
        vi = curr_vertex->position; // nm
        Nvi = curr_vertex->normal;  // unitless

        // projection matrix
        ffscalar_multd(Nvi, dN, NvidN, VECTORSIZE);  // unitless
        fdsubtract3d(vi,NvidN,viNvidN);
        orthogonal_projection_matrix3(Nvi, p);  // unitless

        // Need a three-pass over the neighbors
        // 1. get the radial weights
        r_sum = 0.0; // 1/nm
        j = 0;
        neighbor = (curr_vertex->neighbors)[j];
        while((neighbor!=-1) && (j<NEIGHBORSIZE))
        {
            curr_neighbor = &(halfedges[neighbor]);
            neighbor_vertex = &(vertices[curr_neighbor->vertex]);

            vj = neighbor_vertex->position;  // nm
            ffsubtract3d(vj,vi,dv); // nm
            dv_norm = norm3(dv);  // nm
            // radial weighting
            if (dv_norm > EPSILON)
                r_sum += 1.0/dv_norm;  // 1/nm

            ++j;
            neighbor = (curr_vertex->neighbors)[j];
        }
        n_neighbors = j;  // record the number of neighbors for later passes

        // zero out Mvi
        for (j=0;j<(VECTORSIZE*VECTORSIZE);++j)
            Mvi[j] = 0.0;

        // 2. Compute Mvi
        dareas = 0.0; // nm^2
        areas = 0.0;  // nm^2
        dE_neighbors[i] = 0.0;  // eV/nm
        for(j=0;j<n_neighbors;++j)
        {
            neighbor = (curr_vertex->neighbors)[j];
            curr_neighbor = &(halfedges[neighbor]);
            neighbor_vertex = &(vertices[curr_neighbor->vertex]);
            vj = neighbor_vertex->position;  // nm
            ffsubtract3d(vj,vi,dv); // nm
            subtract3(dv,NvidN,dv_1);  // nm

            dv_norm = norm3(dv);  // nm
            dv_1_norm = norm3(dv_1);  // nm

            // normalized vectors
            if (dv_norm > EPSILON)
                scalar_divide3(dv,dv_norm,dv_hat);  // unitless
            if (dv_1_norm > EPSILON)
                scalar_divide3(dv_1,dv_1_norm,dv_1_hat);  // unitless

            // tangents
            scalar_mult(dv,-1.0,ndv,VECTORSIZE); // nm
            project3(p, ndv, T_theta); // nm^2
            T_theta_norm = norm3(T_theta); // nm^2
            if (T_theta_norm > EPSILON)
                scalar_divide3(T_theta,T_theta_norm,Tij); // unitless
            else
                for (jj=0;jj<VECTORSIZE;++jj) Tij[jj] = 0.0;

            // edge normals subtracted from vertex normals
            // the square root checks are only needed for non-manifold meshes
            Nvidv_hat = SQUARE(fddot3d(Nvi,dv_hat));
            if (Nvidv_hat > 1.0)
                Ni_diff = sqrt(2.0);
            else
                Ni_diff = sqrt(2.0-2.0*sqrt(1.0-Nvidv_hat));  // unitless
            Nvj = neighbor_vertex->normal;  // unitless
            Nvjdv_hat = SQUARE(fddot3d(Nvj,dv_hat));
            if (Nvjdv_hat > 1.0)
                Nj_diff = sqrt(2.0);
            else
                Nj_diff = sqrt(2.0-2.0*sqrt(1.0-Nvjdv_hat));  // unitless
            Nvjdv_1_hat = SQUARE(fddot3d(Nvj,dv_1_hat));
            if (Nvjdv_1_hat > 1.0)
                Nj_1_diff = sqrt(2.0);
            else
                Nj_1_diff = sqrt(2.0-2.0*sqrt(1.0-Nvjdv_1_hat));  // unitless

            // Compute the principal curvatures from the difference in normals (same as difference in tangents)
            kj = safe_divide(2.0*Nj_diff, dv_norm);  // 1/nm
            kj_1 = safe_divide(2.0*Nj_1_diff, dv_1_norm); // 1/nm

            // weights/areas
            w = safe_divide(safe_divide(1.0,dv_norm),r_sum); // unitless
            k = safe_divide(2.0*SIGN(fddot3d(Nvi,dv))*Ni_diff,dv_norm);  // unitless
            Aj = faces[curr_neighbor->face].area;  // nm^2

            // calculate the area curr_neighbor->face after shifting vi by dN
            next_neighbor = &(halfedges[curr_neighbor->next]);
            next_neighbor_vertex = &(vertices[next_neighbor->vertex]);
            vn = next_neighbor_vertex->position;  // nm
            fdsubtract3d(vn,viNvidN,dvn);
            cross3(dv_1,dvn,dv_1dvn);
            dareas += 0.5*norm3(dv_1dvn);
        
            areas += Aj;  // nm^2
            dE_neighbors[i] += Aj*w*kc*(2.0*kj-c0)*(kj_1-kj)/dN;  // eV

            // Construct Mvi
            outer3(Tij,Tij,Mvi_temp);
            scalar_mult(Mvi_temp,w*k,Mvi_temp2,(VECTORSIZE*VECTORSIZE));
            for (jj=0;jj<(VECTORSIZE*VECTORSIZE);++jj)
                Mvi[jj] += Mvi_temp2[jj];
        }
        dareas = dareas - areas;  // calculate local difference in area after shifting dN

        // Interlude: calculate curvature tensor
        compute_curvature_tensor_eig(Mvi, &l1, &l2, v1, v2);

        if isnan(l1) {
            // weird tensor
            k_1 = 0.0; k_2 = 0.0;
            v1[0] = v1[1] = v1[2] = 0.0;
            v2[0] = v2[1] = v2[2] = 0.0;
        } else {

            // principal curvatures (1/nm)
            k_1 = 3.0*l1 - l2;
            k_2 = 3.0*l2 - l1;
        }

        // mean and gaussian curvatures
        H[i] = (PRECISION)(0.5*(k_1+k_2));  // 1/nm
        K[i] = (PRECISION)(k_1*k_2); // 1/nm^2

        // create little m (eigenvector matrix)
        m[0] = v1[0]; m[3] = v1[1]; m[6] = v1[2];
        m[1] = v2[0]; m[4] = v2[1]; m[7] = v2[2];
        // m[2] = Nvi[0]; m[5] = Nvi[1]; m[8] = Nvi[2];  // TODO: we don't need these assignments, skip?

        // since we're operating at a fixed size, zero out A and At so we don't add
        // extra values to our matrix
        for (j=0;j<NEIGHBORSIZE;++j) A[j] = At[j] = AtAinvAt[j] = b[j] = 0.0;
        for (j=NEIGHBORSIZE;j<(2*NEIGHBORSIZE);++j) A[j] = At[j] = AtAinvAt[j] = 0.0;

        // 3. Compute shift
        for (j=0;j<n_neighbors;++j)
        {
            neighbor = (curr_vertex->neighbors)[j];
            curr_neighbor = &(halfedges[neighbor]);
            neighbor_vertex = &(vertices[curr_neighbor->vertex]);
            vj = neighbor_vertex->position;  // nm

            ffsubtract3d(vj,vi,dv); // nm

            // construct a quadratic in the space of T_1 vs. T_2
            A[2*j] = SQUARE(dv[0]*m[0]+dv[1]*m[3]+dv[2]*m[6]);
            A[2*j+1] = SQUARE(dv[0]*m[1]+dv[1]*m[4]+dv[2]*m[7]);

            // Update the equation y-intercept to displace the curve along the normal direction
            b[j] = A[2*j]*k_1+A[2*j+1]*k_2 - (double)dN;
        }

        // solve 
        transpose(A, At, NEIGHBORSIZE, 2);  // construct A transpose
        matmul(At, A, AtA, 2, NEIGHBORSIZE, 2);  // construct AtA
        moore_penrose_2x2(AtA, AtAinv);  // construct inverted matrix
        matmul(AtAinv, At, AtAinvAt, 2, 2, NEIGHBORSIZE);
        matmul(AtAinvAt, b, k_p, 2, NEIGHBORSIZE, 1);  // k_p are principal curvatures after displacement

        dH[i] = (PRECISION)(-1.0*(0.5*(k_p[0] + k_p[1]) - (double)H[i])/((double)dN));  // 1/nm
        dK[i] = (PRECISION)(-1.0*((k_p[0]-k_1)*k_2 + k_1*(k_p[1]-k_2))/((double)dN));  // 1/nm^2

        E[i] = (PRECISION)(areas*((double)(0.5*kc*SQUARE(2.0*H[i] - c0) + kg*K[i])));

        pE[i] = (PRECISION)(exp(-(1.0/KBT)*((double)(E[i]))));

        // Take into account the change in neighboring energies for each vertex shift
        // Compute dEdN by component
        dEdN_H = dareas*((double)kc)*(2.0*((double)(H[i]))-((double)c0))*((double)(dH[i]));  // eV/nm^2
        dEdN_K = dareas*((double)kg)*((double)(dK[i]));  // eV/nm^2
        dEdN_sum = (dEdN_H + dEdN_K + dE_neighbors[i]); // + dE_neighbors[i]); // eV/nm^2 # + dE_neighbors[i])
        dEdNs = (PRECISION)(-1.0*dEdN_sum)*(1.0-pE[i]);  // *(1.0-pE[i]); // eV/nm # *(1.0-pE[i]);  // drive dEdNs toward 0

        // printf("%e %e %e %e %e %e\n", dareas, dH[i], dK[i], dEdN_H, dEdN_K, dEdNs);

        for (jj=0;jj<VECTORSIZE;++jj) {
            (dEdN[i]).position[jj] = dEdNs*Nvi[jj];
            // if isnan((dEdN[i]).position[jj]) {
            //     printf("%e %e\n", l1, l2);
            //     printf("%e %e %d %e %e \n", (dEdN[i]).position[jj], areas, n_neighbors, k_p[0], k_p[1]);
            //     printf("%e %e\n", AtA[0], AtA[1]);
            //     printf("%e %e\n", AtA[2], AtA[3]);
            //     printf("%e %e\n", AtAinv[0], AtAinv[1]);
            //     printf("%e %e\n", AtAinv[2], AtAinv[3]);
            //     for (j=0;j<2*NEIGHBORSIZE;++j)
            //         printf("%e ", A[j]);
            //     printf("\n");
            //     for (j=0;j<NEIGHBORSIZE;++j)
            //         printf("%e ", b[j]);
            //     printf("\n");
            // }
        }
    }
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
 *  @param dE_neighbors PRECISION n_vertices vector of how a shift at this vertex changes the bending energy in 1-ring neighbor vertices
 *  @param kc PRECISION bending stiffness coefficient (eV)
 *  @param kg PRECISION bending stiffness coefficient (eV)
 *  @param c0 PRECISION spontaneous curvature (1/nm)
 *  @param dEdN PRECISION n_vertices vector of energy gradient
 *  @return Void
 */
static void c_curvature_grad_centroid(void *vertices_, 
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
    int i, j, jj, neighbor, n_neighbors;
    double l1, l2, r_sum, dv_norm, dv_1_norm, T_theta_norm, Ni_diff, Nj_diff, Nj_1_diff;
    double kj, kj_1, k, Aj, areas, dareas, w, k_1, k_2;
    double dEdN_H, dEdN_K, dEdN_sum;
    double Nvidv_hat, Nvjdv_hat, Nvjdv_1_hat;
    double v1[VECTORSIZE], v2[VECTORSIZE], Mvi[VECTORSIZE*VECTORSIZE];
    double Mvi_temp[VECTORSIZE*VECTORSIZE], Mvi_temp2[VECTORSIZE*VECTORSIZE];
    double m[VECTORSIZE*VECTORSIZE];
    double p[VECTORSIZE*VECTORSIZE], dv[VECTORSIZE], dvn[VECTORSIZE], dv_1dvn[VECTORSIZE], ndv[VECTORSIZE];
    double dv_hat[VECTORSIZE], dv_1[VECTORSIZE], dv_1_hat[VECTORSIZE];
    double NvidN[VECTORSIZE], viNvidN[VECTORSIZE], T_theta[VECTORSIZE], Tij[VECTORSIZE];
    double A[2*NEIGHBORSIZE], At[2*NEIGHBORSIZE], AtA[4], AtAinv[4], AtAinvAt[2*NEIGHBORSIZE];
    double b[NEIGHBORSIZE], k_p[2];
    PRECISION dEdNs, vj_centroid[3], vivj[3], vivj_norm;
    PRECISION *vi, *vj, *vn, *Nvi, *Nvj;
    vertex_t *curr_vertex, *neighbor_vertex, *next_neighbor_vertex;
    halfedge_t *curr_neighbor, *next_neighbor;
    vertex_t *vertices = (vertex_t*) vertices_;
    face_t *faces = (face_t*) faces_;

    for (i=0;i<n_vertices;++i)
    {
        curr_vertex = &(vertices[i]);
        // Skip unused vertices || stochastically choose which vertices to adjust
        if ( ((curr_vertex->halfedge) == -1) || ((skip_prob>0)&&(r2()<skip_prob)) ) {
            H[i] = 0.0;
            K[i] = 0.0;
            dH[i] = 0.0;
            dK[i] = 0.0;
            dE_neighbors[i] = 0.0;
            E[i] = 0.0;
            pE[i] = 0.0;
            for (jj=0;jj<VECTORSIZE;++jj)
                (dEdN[i]).position[jj] = 0.0;
            continue;
        }

        // Vertex and its normal
        vi = curr_vertex->position; // nm
        Nvi = curr_vertex->normal;  // unitless

        // zero out neighbor centroid
        for (jj=0;jj<3;++jj)
            vj_centroid[jj] = 0.0;

        // Need a three-pass over the neighbors
        // 1. get the radial weights
        r_sum = 0.0; // 1/nm
        j = 0;
        neighbor = (curr_vertex->neighbors)[j];
        while((neighbor!=-1) && (j<NEIGHBORSIZE))
        {
            curr_neighbor = &(halfedges[neighbor]);
            neighbor_vertex = &(vertices[curr_neighbor->vertex]);

            vj = neighbor_vertex->position;  // nm
            // update centroid position
            for (jj=0;jj<3;++jj)
                vj_centroid[jj] += vj[jj];
            ffsubtract3d(vj,vi,dv); // nm
            dv_norm = norm3(dv);  // nm
            // radial weighting
            if (dv_norm > EPSILON)
                r_sum += 1.0/dv_norm;  // 1/nm

            ++j;
            neighbor = (curr_vertex->neighbors)[j];
        }
        n_neighbors = j;  // record the number of neighbors for later passes

        // average the position
        for (jj=0;jj<3;++jj)
            vj_centroid[jj] /= n_neighbors;
        // calculate the normal vector pointing from vi to vj_centroid
        ffsubtract3f(vj_centroid,vi,vivj);
        vivj_norm = fnorm3f(vivj);
        for (jj=0;jj<3;++jj)
            vivj[jj] /= vivj_norm;

        // projection matrix
        ffscalar_multd(vivj, dN, NvidN, VECTORSIZE);  // unitless

        fdsubtract3d(vi,NvidN,viNvidN);
        orthogonal_projection_matrix3(Nvi, p);  // unitless

        // zero out Mvi
        for (j=0;j<(VECTORSIZE*VECTORSIZE);++j)
            Mvi[j] = 0.0;

        // 2. Compute Mvi
        dareas = 0.0; // nm^2
        areas = 0.0;  // nm^2
        dE_neighbors[i] = 0.0;  // eV/nm
        for(j=0;j<n_neighbors;++j)
        {
            neighbor = (curr_vertex->neighbors)[j];
            curr_neighbor = &(halfedges[neighbor]);
            neighbor_vertex = &(vertices[curr_neighbor->vertex]);
            vj = neighbor_vertex->position;  // nm
            ffsubtract3d(vj,vi,dv); // nm
            subtract3(dv,NvidN,dv_1);  // nm

            dv_norm = norm3(dv);  // nm
            dv_1_norm = norm3(dv_1);  // nm

            // normalized vectors
            if (dv_norm > EPSILON)
                scalar_divide3(dv,dv_norm,dv_hat);  // unitless
            if (dv_1_norm > EPSILON)
                scalar_divide3(dv_1,dv_1_norm,dv_1_hat);  // unitless

            // tangents
            scalar_mult(dv,-1.0,ndv,VECTORSIZE); // nm
            project3(p, ndv, T_theta); // nm^2
            T_theta_norm = norm3(T_theta); // nm^2
            if (T_theta_norm > EPSILON)
                scalar_divide3(T_theta,T_theta_norm,Tij); // unitless
            else
                for (jj=0;jj<VECTORSIZE;++jj) Tij[jj] = 0.0;

            // edge normals subtracted from vertex normals
            // the square root checks are only needed for non-manifold meshes
            Nvidv_hat = SQUARE(fddot3d(Nvi,dv_hat));
            if (Nvidv_hat > 1.0)
                Ni_diff = sqrt(2.0);
            else
                Ni_diff = sqrt(2.0-2.0*sqrt(1.0-Nvidv_hat));  // unitless
            Nvj = neighbor_vertex->normal;  // unitless
            Nvjdv_hat = SQUARE(fddot3d(Nvj,dv_hat));
            if (Nvjdv_hat > 1.0)
                Nj_diff = sqrt(2.0);
            else
                Nj_diff = sqrt(2.0-2.0*sqrt(1.0-Nvjdv_hat));  // unitless
            Nvjdv_1_hat = SQUARE(fddot3d(Nvj,dv_1_hat));
            if (Nvjdv_1_hat > 1.0)
                Nj_1_diff = sqrt(2.0);
            else
                Nj_1_diff = sqrt(2.0-2.0*sqrt(1.0-Nvjdv_1_hat));  // unitless

            // Compute the principal curvatures from the difference in normals (same as difference in tangents)
            kj = safe_divide(2.0*Nj_diff, dv_norm);  // 1/nm
            kj_1 = safe_divide(2.0*Nj_1_diff, dv_1_norm); // 1/nm

            // weights/areas
            w = safe_divide(safe_divide(1.0,dv_norm),r_sum); // unitless
            k = safe_divide(2.0*SIGN(fddot3d(Nvi,dv))*Ni_diff,dv_norm);  // unitless
            Aj = faces[curr_neighbor->face].area;  // nm^2

            // calculate the area curr_neighbor->face after shifting vi by dN
            next_neighbor = &(halfedges[curr_neighbor->next]);
            next_neighbor_vertex = &(vertices[next_neighbor->vertex]);
            vn = next_neighbor_vertex->position;  // nm
            fdsubtract3d(vn,viNvidN,dvn);
            cross3(dv_1,dvn,dv_1dvn);
            dareas += 0.5*norm3(dv_1dvn);
        
            areas += Aj;  // nm^2
            dE_neighbors[i] += Aj*w*kc*(2.0*kj-c0)*(kj_1-kj)/dN;  // eV

            // Construct Mvi
            outer3(Tij,Tij,Mvi_temp);
            scalar_mult(Mvi_temp,w*k,Mvi_temp2,(VECTORSIZE*VECTORSIZE));
            for (jj=0;jj<(VECTORSIZE*VECTORSIZE);++jj)
                Mvi[jj] += Mvi_temp2[jj];
        }
        dareas = dareas - areas;  // calculate local difference in area after shifting dN

        // Interlude: calculate curvature tensor
        compute_curvature_tensor_eig(Mvi, &l1, &l2, v1, v2);

        if isnan(l1) {
            // weird tensor
            k_1 = 0.0; k_2 = 0.0;
            v1[0] = v1[1] = v1[2] = 0.0;
            v2[0] = v2[1] = v2[2] = 0.0;
        } else {

            // principal curvatures (1/nm)
            k_1 = 3.0*l1 - l2;
            k_2 = 3.0*l2 - l1;
        }

        // mean and gaussian curvatures
        H[i] = (PRECISION)(0.5*(k_1+k_2));  // 1/nm
        K[i] = (PRECISION)(k_1*k_2); // 1/nm^2

        // create little m (eigenvector matrix)
        m[0] = v1[0]; m[3] = v1[1]; m[6] = v1[2];
        m[1] = v2[0]; m[4] = v2[1]; m[7] = v2[2];
        // m[2] = Nvi[0]; m[5] = Nvi[1]; m[8] = Nvi[2];  // TODO: we don't need these assignments, skip?

        // since we're operating at a fixed size, zero out A and At so we don't add
        // extra values to our matrix
        for (j=0;j<NEIGHBORSIZE;++j) A[j] = At[j] = AtAinvAt[j] = b[j] = 0.0;
        for (j=NEIGHBORSIZE;j<(2*NEIGHBORSIZE);++j) A[j] = At[j] = AtAinvAt[j] = 0.0;

        // 3. Compute shift
        for (j=0;j<n_neighbors;++j)
        {
            neighbor = (curr_vertex->neighbors)[j];
            curr_neighbor = &(halfedges[neighbor]);
            neighbor_vertex = &(vertices[curr_neighbor->vertex]);
            vj = neighbor_vertex->position;  // nm

            ffsubtract3d(vj,vi,dv); // nm

            // construct a quadratic in the space of T_1 vs. T_2
            A[2*j] = SQUARE(dv[0]*m[0]+dv[1]*m[3]+dv[2]*m[6]);
            A[2*j+1] = SQUARE(dv[0]*m[1]+dv[1]*m[4]+dv[2]*m[7]);

            // Update the equation y-intercept to displace the curve along the normal direction
            b[j] = A[2*j]*k_1+A[2*j+1]*k_2 - (double)dN;
        }

        // solve 
        transpose(A, At, NEIGHBORSIZE, 2);  // construct A transpose
        matmul(At, A, AtA, 2, NEIGHBORSIZE, 2);  // construct AtA
        moore_penrose_2x2(AtA, AtAinv);  // construct inverted matrix
        matmul(AtAinv, At, AtAinvAt, 2, 2, NEIGHBORSIZE);
        matmul(AtAinvAt, b, k_p, 2, NEIGHBORSIZE, 1);  // k_p are principal curvatures after displacement

        dH[i] = (PRECISION)(-1.0*(0.5*(k_p[0] + k_p[1]) - (double)H[i])/((double)dN));  // 1/nm
        dK[i] = (PRECISION)(-1.0*((k_p[0]-k_1)*k_2 + k_1*(k_p[1]-k_2))/((double)dN));  // 1/nm^2

        E[i] = (PRECISION)(areas*((double)(0.5*kc*SQUARE(2.0*H[i] - c0) + kg*K[i])));

        pE[i] = (PRECISION)(exp(-(1.0/KBT)*((double)(E[i]))));

        // Take into account the change in neighboring energies for each vertex shift
        // Compute dEdN by component
        dEdN_H = dareas*((double)kc)*(2.0*((double)(H[i]))-((double)c0))*((double)(dH[i]));  // eV/nm^2
        dEdN_K = dareas*((double)kg)*((double)(dK[i]));  // eV/nm^2
        dEdN_sum = (dEdN_H + dEdN_K + dE_neighbors[i]); // + dE_neighbors[i]); // eV/nm^2 # + dE_neighbors[i])
        dEdNs = (PRECISION)(-1.0*dEdN_sum)*(1.0-pE[i]);  // *(1.0-pE[i]); // eV/nm # *(1.0-pE[i]);  // drive dEdNs toward 0

        // printf("%e %e %e %e %e %e\n", dareas, dH[i], dK[i], dEdN_H, dEdN_K, dEdNs);

        for (jj=0;jj<VECTORSIZE;++jj) {
            (dEdN[i]).position[jj] = dEdNs*vivj[jj];
            // if isnan((dEdN[i]).position[jj]) {
            //     printf("%e %e\n", l1, l2);
            //     printf("%e %e %d %e %e \n", (dEdN[i]).position[jj], areas, n_neighbors, k_p[0], k_p[1]);
            //     printf("%e %e\n", AtA[0], AtA[1]);
            //     printf("%e %e\n", AtA[2], AtA[3]);
            //     printf("%e %e\n", AtAinv[0], AtAinv[1]);
            //     printf("%e %e\n", AtAinv[2], AtAinv[3]);
            //     for (j=0;j<2*NEIGHBORSIZE;++j)
            //         printf("%e ", A[j]);
            //     printf("\n");
            //     for (j=0;j<NEIGHBORSIZE;++j)
            //         printf("%e ", b[j]);
            //     printf("\n");
            // }
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