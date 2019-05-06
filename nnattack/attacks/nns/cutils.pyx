#cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def check_feasibility(double[:, :] G_view, double[:] h_view, double[:] x_view, int n, int m):
    # True if feasible
    cdef int i, j, ret = 1
    cdef double d
    for i in range(n):
        d = -h_view[i]
        for j in range(m):
            d += G_view[i, j] * x_view[j]
        if d >= 0:
            print(i, d)
        ret = ret & (d < 0)
    return ret

def c_get_half_space(np.ndarray[DTYPE_t, ndim=1] a, np.ndarray[DTYPE_t, ndim=1] b):
    cdef int i, j
    cdef double wa, wb, c
    cdef int n = a.shape[0]
    cdef np.ndarray w = np.zeros(n+1, dtype=DTYPE)
    cdef double [:] a_view = a
    cdef double [:] b_view = b
    cdef double [:] w_view = w

    for i in range(n):
        w_view[i] = a_view[i] - b_view[i]
        wa += w_view[i] * a_view[i]
        wb += w_view[i] * b_view[i]

    c = (wa + wb) / 2
    w_view[n] = c

    if wb > c:
        return -w
    else:
        return w

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef double c_l2_distance(double[:] w_view, double[:] x_view, double[:] t_view, int n) nogil:
    cdef int i
    cdef double ret = 0

    for i in range(n):
        ret += (t_view[i] - x_view[i]) * w_view[i]
    return ret

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void cx_get_half_space(
    double[:] a_view, double[:] b_view, double[:] w_view, int m) nogil:
    cdef int i, j
    cdef double wa=0, wb=0, c

    for i in range(m):
        w_view[i] = a_view[i] - b_view[i]
        wa += w_view[i] * a_view[i]
        wb += w_view[i] * b_view[i]

    c = (wa + wb) / 2
    w_view[m] = c
    if wb > c:
        for i in range(m+1):
            w_view[i] *= -1

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def get_all_half_spaces(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=1] Xj):
    cdef int i, j
    cdef int n = X.shape[0], m = X.shape[1]
    cdef np.ndarray G = np.zeros((n, m), dtype=DTYPE)
    cdef np.ndarray h = np.zeros(n, dtype=DTYPE)
    cdef np.ndarray w = np.zeros(m+1, dtype=DTYPE)
    cdef double [:, :] G_view = G
    cdef double [:] h_view = h
    cdef double [:, :] X_view = X
    cdef double [:] w_view = w

    for i in range(n):
        cx_get_half_space(X_view[i], Xj, w_view, m)
        with nogil:
            for j in range(m):
                G_view[i, j] = w_view[j]
            #G[i] = w[:-1]
            h_view[i] = w_view[m]

    return G, h

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def get_constraints(np.ndarray[DTYPE_t, ndim=2] trnX, np.ndarray tuple_x,
        kdtree, int faropp, np.ndarray[DTYPE_t, ndim=1] tar_x):
    cdef int i, j, k, n_points
    cdef int n_tuples = len(tuple_x)
    cdef int n = trnX.shape[0], m = trnX.shape[1]
    cdef np.ndarray[np.int_t, ndim=1] ps = np.zeros(n, dtype=np.int)
    cdef np.ndarray[DTYPE_t, ndim=2] G
    cdef np.ndarray[DTYPE_t, ndim=1] h
    cdef np.ndarray[DTYPE_t, ndim=1] w = np.zeros(m+1, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] dist
    cdef double [:, :] G_view
    cdef double [:] dist_view
    cdef double [:] h_view
    cdef double [:] tx_view = tar_x
    cdef double [:] w_view = w
    cdef double [:, :] trnX_view = trnX
    cdef long [:] points_view = ps
    cdef long [:] tuple_view = tuple_x
    cdef long pi

    Gl, hl, dl = [], [], []
    for j in range(n_tuples):
        j = tuple_view[j]
        if faropp == -1:
            #points = np.asarray([i for i in range(len(trnX)) if i not in tuple_x])
            with nogil:
                n_points = 0
                for i in range(n):
                    for k in range(n_tuples):
                        if i == tuple_view[k]:
                            k = -1
                            break
                    if k != -1:
                        points_view[n_points] = i
                        n_points += 1
        else:
            near_points = kdtree.query(trnX[j].reshape((1, -1)),
                                       k=min(faropp, len(trnX)),
                                       return_distance=False)
            points = np.asarray([i for i in near_points[0] if i not in tuple_x])
            n_points = len(points)
            ps[:n_points] = points

        G = np.zeros((n_points, m), dtype=DTYPE)
        h = np.empty(n_points, dtype=DTYPE)
        dist = np.empty(n_points, dtype=DTYPE)
        G_view, h_view, dist_view = G, h, dist

        for i in range(n_points):
            pi = points_view[i]
            cx_get_half_space(trnX_view[pi], trnX_view[j], w_view, m)
            for k in range(m):
                G_view[i, k] = w_view[k]
            h_view[i] = w_view[m]
            dist_view[i] = c_l2_distance(w_view, trnX_view[pi], tx_view, m)

        Gl.append(G)
        hl.append(h)
        dl.append(dist)

    #G, h = matrix(np.vstack(G), tc='d'), matrix(np.concatenate(h), tc='d')
    G, h, d = np.vstack(Gl), np.concatenate(hl), np.concatenate(dl)
    return G, h, d
