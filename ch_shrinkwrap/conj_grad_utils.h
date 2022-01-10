#ifndef _conj_grad_utils_h_
#define _conj_grad_utils_h_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

static PyObject *c_shrinkwrap_a_func(PyObject *self, PyObject *args);
static PyObject *c_shrinkwrap_ah_func(PyObject *self, PyObject *args);
static PyObject *c_compute_weight_matrix(PyObject *self, PyObject *args);
static PyObject *c_shrinkwrap_l_func(PyObject *self, PyObject *args);
static PyObject *c_shrinkwrap_lh_func(PyObject *self, PyObject *args);

#ifdef __cplusplus
}
#endif

#endif /* _conj_grad_utils_h_ */