import numpy as np
import scipy.sparse as sparse

within_connectivity = 0.2
outwith_connectivity = 0.05

total_size = 6
inner_size = 3
n_sub_reservoirs = 2
np.random.seed(1)

def create_random_esn_weights(total_size, density=within_connectivity):
    W = sparse.random(total_size, total_size, density)
    W.data = (W.data - 0.5) * 2
    W = W.toarray()
    s = np.linalg.svd(W, compute_uv=False)
    W = W / s[0]
    return W

def create_restricted_esn_weights(total_size, inner_size, n_sub_reservoirs, within_connectivity=within_connectivity, outwith_connectivity=outwith_connectivity, svd_dv=1):

    if total_size != inner_size*n_sub_reservoirs:
        raise ValueError("total size must be the number of reservoirs*their size")

    W = sparse.random(total_size, total_size, outwith_connectivity)
    W.data = (W.data - 0.5) * 2
    W = W.toarray()

    for i in range(n_sub_reservoirs):
        W_inner = sparse.random(inner_size, inner_size, within_connectivity)
        W_inner.data = (W_inner.data - 0.5) * 2
        W_inner = W_inner.toarray()
        indices = np.array(range(inner_size))+(i*inner_size)
        # print(indices)
        coords = np.meshgrid(indices, indices)
        # print(coords)
        W[tuple(coords)] = W_inner
    s = np.linalg.svd(W, compute_uv=False)
    W = W / (s[0]/svd_dv)
    return W

def zero_Wn(W, Wn_size, n_index):
    '''
    Set the within matrix to identity
    
    W: full weight matrix
    Wn_size: size of the sub-reservoir
    n_index: index of the reservoir to zero

    example:
    original_matrix:
    [[a, a, x, x],
     [a, a, x, x],
     [x, x, b, b],
     [x, x, b, b]]
     new_matrix = zero_Wn(original_matrix, 2, 1)
    new_matrix:
    [[a, a, x, x],
     [a, a, x, x],
     [x, x, 1, 0],
     [x, x, 0, 1]]
    '''
    W_new = np.copy(W)
    id = np.identity(Wn_size)
    indices = np.asarray(range(Wn_size)) + (Wn_size*n_index)
    coords = np.meshgrid(indices, indices)
    W_new[tuple(coords)] = id
    return W_new

def zero_On_all(W, Wn_size, n_index):
    '''
    Set all the values in Wn's rows (excluding Wn itself) to 0.

    This is equivalent to removing all the incoming edges to the submatrix
    Wn_size: size of the subreservoir Wn
    n_index: index of the subreservoir in question

    example:
    original_matrix:
    [[a, a, x, x],
     [a, a, x, x],
     [x, x, b, b],
     [x, x, b, b]]
    new_matrix = zero_On_all(original_matrix, 2, 1)
    new_matrix: 
    [[a, a, x, x],
     [a, a, x, x],
     [0, 0, b, b],
     [0, 0, b, b]]
    '''
    W_new = np.copy(W)
    zeros = np.zeros((W.shape[1], Wn_size))
    indices_Wn = np.asarray(range(Wn_size)) + (Wn_size*n_index)
    coords_Wn = np.meshgrid(indices_Wn, indices_Wn)
    Wn = np.asarray(W[tuple(coords_Wn)])
    # print(Wn)
    indices_zeros = np.asarray(range(W.shape[1]))
    coords_zeros = np.meshgrid(indices_Wn, indices_zeros)
    W_new[tuple(coords_zeros)] = zeros
    W_new[tuple(coords_Wn)] = Wn
    return W_new

def zero_all_On(W, Wn_size, n_index):
    '''
    Set all the values in Wn's columns (excluding Wn itself) to 0.

    This is equivalent to removing all the outgoing edges of the submatrix
    Wn_size: size of the subreservoir Wn
    n_index: index of the subreservoir in question

    example:
    original_matrix:
    [[a, a, x, x],
     [a, a, x, x],
     [x, x, b, b],
     [x, x, b, b]]
    new_matrix = zero_On_all(original_matrix, 2, 1)
    new_matrix: 
    [[a, a, 0, 0],
     [a, a, 0, 0],
     [x, x, b, b],
     [x, x, b, b]]
    '''
    W_new = np.copy(W)
    zeros = np.zeros((Wn_size, W.shape[1]))
    indices_Wn = np.asarray(range(Wn_size)) + (Wn_size*n_index)
    coords_Wn = np.meshgrid(indices_Wn, indices_Wn)
    Wn = np.asarray(W[tuple(coords_Wn)])
    # print(Wn)
    indices_zeros = np.asarray(range(W.shape[1]))
    coords_zeros = np.meshgrid(indices_zeros, indices_Wn)
    W_new[tuple(coords_zeros)] = zeros
    W_new[tuple(coords_Wn)] = Wn
    return W_new

def create_restricted_esn_input_weights(N, K):
    return np.random.uniform(-1, 1, size = (N, K))

def zero_Un(U, Wn_size, n_index):
    """
    Zero the inputs to one of the subreservoirs
    Wn_size: size of the subreservoir Wn
    n_index: index of the subreservoir in question

    example: 
    original_U: 
    [[a, a],
     [a, a],
     [b, b],
     [b, b]]
    new_U = zero_Un(original_U, 2, 1)
    new_U: 
    [[a, a],
     [a, a],
     [0, 0],
     [0, 0]]
    """
    U_new = np.copy(U)
    indices_x = np.asarray(range(Wn_size)) + (Wn_size*n_index)
    indices_y = np.asarray(range(U.shape[1]))
    zeros = np.zeros((U.shape[1], Wn_size))
    coords = np.meshgrid(indices_x, indices_y)
    U_new[tuple(coords)] = zeros
    return U_new

# W = create_restricted_esn_weights(total_size, inner_size, n_sub_reservoirs)
# print(W)
# print("ye")
# W_zeroed = zero_Wn(W, inner_size, 1)
# U_initial = create_restricted_esn_input_weights(total_size, 1)
# print(U_initial)
# U_zeroed = zero_Un(U_initial, inner_size, 1)
# print(U_zeroed)
# print(W_zeroed)