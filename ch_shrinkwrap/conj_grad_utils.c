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
            w = *((float *)PyArray_GETPTR2(v_w, i, k));
            for (j=0; j<n_dims; ++j)
            {
                // p_d[k*n_dims+j] += p_f[i*n_dims+j]*(p_w[i*n_points+k]);
                
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
            w = *((float *)PyArray_GETPTR2(v_w, i, k));
            for (j=0; j<n_dims; ++j)
            {
                // p_d[i*n_dims+j] += p_f[k*n_dims+j]*(p_w[i*n_points+k]);
                
                p_d[i*n_dims+j] += p_f[k*n_dims+j]*w;
            }
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}

void compute_weight_matrix_3D(PyObject* v_f, PyObject* v_n, PyObject* v_points, PyObject* v_dd, int n_verts, int n_points, float shield_sigma)
{
    int i, j, k;
    float * pt3;
    float * pfi;
    float * p_f;
    float *ds;
    float dd_ik, dik, ss2, w; 
    #define N_DIMS 3 //use a fixed constant to allow loop-unrolling and/or vectorisation
    
    // replace with python/numpy allocation (typically faster as from pre-allocated heap) 
    //ds = (float *) calloc(n_points, sizeof(float));
    ds = (float *) PyMem_Calloc(n_points, sizeof(float));

    //initialise to zero
    for (k=0; k<n_points; ++k)
    {
        ds[k] = 0;
    }

    ss2 = 2*shield_sigma*shield_sigma;

    p_f = (float *)PyArray_GETPTR1(v_f, 0);

    for (i=0; i<n_verts; ++i)
    {
        if ((*((float *)PyArray_GETPTR2(v_n, i, 0))) == -1) continue;

        pfi = p_f + i*N_DIMS;

        for (k=0; k<n_points; ++k)
        {
            w = 0;
            pt3 = (float *)PyArray_GETPTR2(v_points, k, 0);

            dd_ik = 0; //*((float *)PyArray_GETPTR2(v_dd, i, k)) 
            
            for (j=0; j<N_DIMS; ++j)
            {
                dik = pfi[j] - pt3[j];
                dd_ik += dik*dik;
            }

            // update to point attraction force
            if (dd_ik > 0) {
                // *((float *)PyArray_GETPTR2(v_dd, i, k)) = (1.0/dik)*exp(-dik/ss2);
                //w = expf(-dd_ik/ss2); // use fexp to avoid casts
                //dd_ik = (1.0/dd_ik)*w; 
                dd_ik = (1.0/sqrtf(dd_ik));
                *((float *)PyArray_GETPTR2(v_dd, i, k)) = dd_ik; 
            }

            // keep track of sum along n_verts
            ds[k] += dd_ik;
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

    //free(ds);
    PyMem_Free(ds);

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

    if (n_dims == 3){
        // use fixed size code
        compute_weight_matrix_3D(v_f, v_n, v_points, v_dd, n_verts, n_points, shield_sigma);
        Py_INCREF(Py_None);
        return Py_None; 
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
                // *((float *)PyArray_GETPTR2(v_dd, i, k)) = (1.0/dik)*exp(-dik/ss2);
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