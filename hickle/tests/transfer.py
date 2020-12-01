    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    sm1 = csr_matrix((data, (row, col)), shape=(3, 3))
    sm2 = csc_matrix((data, (row, col)), shape=(3, 3))

    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
    sm3 = bsr_matrix((data, indices, indptr), shape=(6, 6))

    hickle.dump(sm1, 'test_sp.h5')
    sm1_h = hickle.load('test_sp.h5')
    hickle.dump(sm2, 'test_sp2.h5')
    sm2_h = hickle.load('test_sp2.h5')
    hickle.dump(sm3, 'test_sp3.h5')
    sm3_h = hickle.load('test_sp3.h5')
