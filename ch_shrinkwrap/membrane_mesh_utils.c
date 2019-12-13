#include "Python.h"
#include <math.h>
#include "numpy/arrayobject.h"

#include "membrane_mesh_utils.h"

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