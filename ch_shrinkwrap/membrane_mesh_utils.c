#include <stdio.h>

#include "Python.h"
#include <math.h>
#include "numpy/arrayobject.h"

#include "membrane_mesh_utils.h"

float norm(const float *pos)
{
    float n = 0;
    int i = 0;

    for (i = 0; i < VECTORSIZE; ++i)
        n += pos[i] * pos[i];
    return sqrt(n);
}

float safe_divide(float x, float y)
{
    if (y==0)
        return 0;
    return x/y;
}

void scalar_divide(const float *a, const float b, float *d)
{
    int k = 0;
    for (k=0; k < 3; ++k)
        d[k] = a[k]/b;
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

/*
calc_pt_weight_matrix -- calculate point weight matrix and point weights
for point attraction force
*/
static void calc_pt_weight_matrix(points_t *points, 
                                  void *vertices_, 
                                  float *pt_weight_matrix, 
                                  float *pt_weights, 
                                  float w, 
                                  float charge_sigma, 
                                  int n_points, 
                                  int n_vertices)
{
    int i, j, k;
    float charge_sigma_2, tmp, tmp_diff;
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

static void c_point_attraction_grad(points_t *attraction, 
                             points_t *points, 
                             float *sigma, 
                             vertex_t *vertices, 
                             float w, 
                             float charge_sigma, 
                             int n_points, 
                             int n_vertices)
{
    int i, j, k;
    int32_t curr_idx;
    float *pt_weight_matrix;
    float *pt_weights;
    vertex_t *curr_vertex;
    points_t *curr_point, *curr_attraction;
    float d[VECTORSIZE];
    float dd, r, r2, r12, rf;

    pt_weight_matrix = (float *)malloc(sizeof(float)*n_points*n_vertices);  // n_points x n_vertices
    pt_weights = (float *)malloc(sizeof(float)*n_points); // n_points
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

/*
compute_curvature_tensor_eig -- find eigenvalues/vectors of 3x3 curvature
tensor using closed-form solution
*/
static void compute_curvature_tensor_eig(float *Mvi, float *l1, float *l2, float *v1, float *v2) 
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
    z1 = safe_divide(z1n, z1d);
    y1n = (m12*z1 + m01);
    y1d = (m11 - (*l1));
    y1 = safe_divide(y1n, y1d);
    
    v1t[0] = 1; v1t[1] = y1; v1t[2] = z1;
    scalar_divide(v1t,norm(v1t),v1);
    
    z2n = ((m00 - (*l2))*(m11 - (*l2)) - (m01*m01));
    z2d = (m01*m12 - m02*(m11 - (*l2)));
    z2 = safe_divide(z2n, z2d);
    y2n = (m12*z2 + m01);
    y2d = (m11 - (*l2));
    y2 = safe_divide(y2n, y2d);
    
    v2t[0] = 1; v2t[1] = y2; v2t[2] = z2;
    scalar_divide(v2t,norm(v2t),v2);
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