#include "Python.h"
#include <math.h>
#include <stdlib.h>
#include "numpy/arrayobject.h"

#include "conj_grad_utils.h"

static PyObject *c_shrinkwrap_a_func(PyObject *self, PyObject *args)
{
    PyObject *v_f = 0, *v_n = 0, *v_w = 0, *v_d = 0;
    int n_dims, n_points, n_verts, n_n, i, j, k;
    float w;
    float *p_f;
    // float *p_n;
    // float *p_w;
    float *p_d;

    if (!PyArg_ParseTuple(args, "OOOOiiii", &v_f, &v_n, &v_w, &v_d, &n_dims, &n_points, &n_verts, &n_n)) return NULL;
    if (!PyArray_Check(v_f) || !PyArray_ISCONTIGUOUS(v_f))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the f data.");
        return NULL;
    }
    if (!PyArray_Check(v_n) || !PyArray_ISCONTIGUOUS(v_n))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the neighbor data.");
        return NULL;
    }
    if (!PyArray_Check(v_w) || !PyArray_ISCONTIGUOUS(v_w)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the weigh matrix.");
        return NULL;
    } 
    if (!PyArray_Check(v_d) || !PyArray_ISCONTIGUOUS(v_d)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the output data.");
        return NULL;
    } 

    // printf("c %d %d %d %d\n", n_dims, n_points, n_verts, n_n);

    p_f = (float *)PyArray_GETPTR1(v_f, 0);
    // p_n = (float *)PyArray_GETPTR2(v_n, 0, 0);
    // p_w = (float *)PyArray_GETPTR2(v_w, 0, 0);
    p_d = (float *)PyArray_GETPTR1(v_d, 0);

    for (i=0; i<n_verts; ++i)
    {
        if ((*((int32_t *)PyArray_GETPTR2(v_n, i, 0))) == -1) continue;
        for (k=0; k<n_points; ++k)
        {
            for (j=0; j<n_dims; ++j)
            {
                // p_d[k*n_dims+j] += p_f[i*n_dims+j]*(p_w[i*n_points+k]);
                w = *((float *)PyArray_GETPTR2(v_w, i, k));
                p_d[k*n_dims+j] += p_f[i*n_dims+j]*w;
            }
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *c_shrinkwrap_ah_func(PyObject *self, PyObject *args)
{
    PyObject *v_f = 0, *v_n = 0, *v_w = 0, *v_d = 0;
    int n_dims, n_points, n_verts, n_n, i, j, k;
    float w;
    float *p_f;
    // float *p_n;
    // float *p_w;
    float *p_d;


    if (!PyArg_ParseTuple(args, "OOOOiiii", &v_f, &v_n, &v_w, &v_d, &n_dims, &n_points, &n_verts, &n_n)) return NULL;
    if (!PyArray_Check(v_f) || !PyArray_ISCONTIGUOUS(v_f))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the f data.");
        return NULL;
    }
    if (!PyArray_Check(v_n) || !PyArray_ISCONTIGUOUS(v_n))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the neighbor data.");
        return NULL;
    }
    if (!PyArray_Check(v_w) || !PyArray_ISCONTIGUOUS(v_w)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the weigh matrix.");
        return NULL;
    } 
    if (!PyArray_Check(v_d) || !PyArray_ISCONTIGUOUS(v_d)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the output data.");
        return NULL;
    } 

    p_f = (float *)PyArray_GETPTR1(v_f, 0);
    // p_n = (float *)PyArray_GETPTR2(v_n, 0, 0);
    // p_w = (float *)PyArray_GETPTR2(v_w, 0, 0);
    p_d = (float *)PyArray_GETPTR1(v_d, 0);

    for (i=0; i<n_verts; ++i)
    {
        if ((*((int32_t *)PyArray_GETPTR2(v_n, i, 0))) == -1) continue;
        for (k=0; k<n_points; ++k)
        {
            for (j=0; j<n_dims; ++j)
            {
                // p_d[i*n_dims+j] += p_f[k*n_dims+j]*(p_w[i*n_points+k]);
                w = *((float *)PyArray_GETPTR2(v_w, i, k));
                p_d[i*n_dims+j] += p_f[k*n_dims+j]*w;
            }
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *c_compute_weight_matrix(PyObject *self, PyObject *args)
{
    PyObject *v_f = 0, *v_n = 0, *v_points = 0, *v_dd = 0;
    int n_dims, n_points, n_verts, n_n, i, j, k;
    float pt, dik, dik2, shield_sigma, ss2, search_rad, rad2, sr2;
    float *p_f;
    float *ds;

    if (!PyArg_ParseTuple(args, "OOOOiiiiff", &v_f, &v_n, &v_points, &v_dd, &n_dims, &n_points, &n_verts, &n_n, &shield_sigma, &search_rad)) return NULL;
    if (!PyArray_Check(v_f) || !PyArray_ISCONTIGUOUS(v_f))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the f data.");
        return NULL;
    }
    if (!PyArray_Check(v_n) || !PyArray_ISCONTIGUOUS(v_n))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the neighbor data.");
        return NULL;
    }
    if (!PyArray_Check(v_points) || !PyArray_ISCONTIGUOUS(v_points))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the point data.");
        return NULL;
    }
    if (!PyArray_Check(v_dd) || !PyArray_ISCONTIGUOUS(v_dd)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the output data.");
        return NULL;
    } 

    p_f = (float *)PyArray_GETPTR1(v_f, 0);

    ds = (float *) calloc(n_points, sizeof(float));

    ss2 = 2*shield_sigma*shield_sigma;

    for (i=0; i<n_verts; ++i)
    {
        if ((*((float *)PyArray_GETPTR2(v_n, i, 0))) == -1) continue;
        for (k=0; k<n_points; ++k)
        {
            for (j=0; j<n_dims; ++j)
            {
                // p_d[k*n_dims+j] += p_f[i*n_dims+j]*(p_w[i*n_points+k]);
                pt = *((float *)PyArray_GETPTR2(v_points, k, j));
                dik = p_f[i*n_dims+j] - pt;
                dik2 = dik*dik;
                *((float *)PyArray_GETPTR2(v_dd, i, k)) += dik2;
            }

            // update to point attraction force
            dik = *((float *)PyArray_GETPTR2(v_dd, i, k));
            if (dik > 0) {
                *((float *)PyArray_GETPTR2(v_dd, i, k)) = (1.0/dik)*exp(-dik/ss2);
            }

            // keep track of sum along n_verts
            ds[k] += *((float *)PyArray_GETPTR2(v_dd, i, k));
        }
    }

    // normalize
    for (i=0; i<n_verts; ++i)
    {
        if ((*((float *)PyArray_GETPTR2(v_n, i, 0))) == -1) continue;
        for (k=0; k<n_points; ++k)
        {
            if (ds[k] == 0) continue;
            *((float *)PyArray_GETPTR2(v_dd, i, k)) /= ds[k];
        }
    }

    free(ds);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *c_shrinkwrap_l_func(PyObject *self, PyObject *args)
{
    PyObject *v_f = 0, *v_n = 0, *v_w = 0, *v_d = 0;
    int n_dims, n_points, n_verts, n_n, i, j, k, N;
    int32_t n;
    float *p_f;
    // float *p_n;
    // float *p_w;
    float *p_d;

    if (!PyArg_ParseTuple(args, "OOOOiiii", &v_f, &v_n, &v_w, &v_d, &n_dims, &n_points, &n_verts, &n_n)) return NULL;
    if (!PyArray_Check(v_f) || !PyArray_ISCONTIGUOUS(v_f))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the f data.");
        return NULL;
    }
    if (!PyArray_Check(v_n) || !PyArray_ISCONTIGUOUS(v_n))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the neighbor data.");
        return NULL;
    }
    if (!PyArray_Check(v_w) || !PyArray_ISCONTIGUOUS(v_w)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the weigh matrix.");
        return NULL;
    } 
    if (!PyArray_Check(v_d) || !PyArray_ISCONTIGUOUS(v_d)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the output data.");
        return NULL;
    } 


    p_f = (float *)PyArray_GETPTR1(v_f, 0);
    p_d = (float *)PyArray_GETPTR1(v_d, 0);

    for (i=0; i<n_verts; ++i)
    {
        if ((*((int32_t *)PyArray_GETPTR2(v_n, i, 0))) == -1) continue;
        for (j=0; j<n_dims; ++j)
        {
            N = 0;
            for (k=0; k<n_n; ++k)
            {
                n = *((int32_t *)PyArray_GETPTR2(v_n, i, k));
                if (n == -1) break;
                p_d[i*n_dims+j] += (p_f[n*n_dims+j] - p_f[i*n_dims+j]);
                N += 1;
            }
            // normalize
            p_d[i*n_dims+j] /= N;
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *c_shrinkwrap_lh_func(PyObject *self, PyObject *args)
{
    PyObject *v_f = 0, *v_n = 0, *v_w = 0, *v_d = 0;
    int n_dims, n_points, n_verts, n_n, i, j, k, N;
    int32_t n;
    float *p_f;
    // float *p_n;
    // float *p_w;
    float *p_d;

    if (!PyArg_ParseTuple(args, "OOOOiiii", &v_f, &v_n, &v_w, &v_d, &n_dims, &n_points, &n_verts, &n_n)) return NULL;
    if (!PyArray_Check(v_f) || !PyArray_ISCONTIGUOUS(v_f))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the f data.");
        return NULL;
    }
    if (!PyArray_Check(v_n) || !PyArray_ISCONTIGUOUS(v_n))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the neighbor data.");
        return NULL;
    }
    if (!PyArray_Check(v_w) || !PyArray_ISCONTIGUOUS(v_w)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the weigh matrix.");
        return NULL;
    } 
    if (!PyArray_Check(v_d) || !PyArray_ISCONTIGUOUS(v_d)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the output data.");
        return NULL;
    } 


    p_f = (float *)PyArray_GETPTR1(v_f, 0);
    p_d = (float *)PyArray_GETPTR1(v_d, 0);

    for (i=0; i<n_verts; ++i)
    {
        if ((*((int32_t *)PyArray_GETPTR2(v_n, i, 0))) == -1) continue;
        for (j=0; j<n_dims; ++j)
        {
            N = 0;
            for (k=0; k<n_n; ++k)
            {
                n = *((int32_t *)PyArray_GETPTR2(v_n, i, k));
                if (n == -1) break;
                p_d[n*n_dims+j] += (p_f[i*n_dims+j] - p_f[n*n_dims+j]);
                N += 1;
            }
            // normalize
            for (k=0; k<N; ++k)
            {
                n = *((int32_t *)PyArray_GETPTR2(v_n, i, k));
                p_d[n*n_dims+j] /= N;
            }
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}


static PyMethodDef conj_grad_utils_methods[] = {
    {"c_shrinkwrap_a_func", c_shrinkwrap_a_func, METH_VARARGS},
    {"c_shrinkwrap_ah_func", c_shrinkwrap_ah_func, METH_VARARGS},
    {"c_compute_weight_matrix", c_compute_weight_matrix, METH_VARARGS},
    {"c_shrinkwrap_l_func", c_shrinkwrap_l_func, METH_VARARGS},
    {"c_shrinkwrap_lh_func", c_shrinkwrap_lh_func, METH_VARARGS},
    {NULL, NULL, 0}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef3 = {
        PyModuleDef_HEAD_INIT,
        "conj_grad_utils",     /* m_name */
        "C implementations of conj_grad operations for speed improvement",  /* m_doc */
        -1,                  /* m_size */
        conj_grad_utils_methods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };

PyMODINIT_FUNC PyInit_conj_grad_utils(void)
{
	PyObject *m;
    m = PyModule_Create(&moduledef3);
    import_array()
    return m;
}
#else
PyMODINIT_FUNC initconj_grad_utils(void)
{
    PyObject *m;

    m = Py_InitModule("conj_grad_utils", conj_grad_utils_methods);
    import_array()

}
#endif