
import numpy as np
from scipy.special.orthogonal import p_roots
from scipy.optimize import linprog
from empirical_cubature_method import EmpiricalCubatureMethod
from randomized_singular_value_decomposition import RandomizedSingularValueDecomposition
from plot_scripts import plot_original_function_1D, plot_basis_functions_1D, plot_sparsity, plot_lp_sparsity,plot_ground_truth




def gauss_quad(M):
    min_ = 0
    max_ = 1
    [X, W] = p_roots(M)
    dx = max_- min_
    dxi = 2
    W = W*(dx/dxi)
    X = ((max_ - min_)/2)*X + ((max_ + min_)/2)

    return X, W




def get_list_of_arrays_sizes(list_of_np_arrays):
    size_1 = 0
    size_2 = 0
    for i in range(len(list_of_np_arrays)):
        size_1+= list_of_np_arrays[i].shape[0]
        size_2+= list_of_np_arrays[i].shape[1]
    return size_1, size_2




def list_to_mat(list_of_np_arrays):
    size_1, size_2= get_list_of_arrays_sizes(list_of_np_arrays)
    numpy_arr = np.zeros((size_1,size_2))

    gap_horizontal = 0
    gap_vertical = 0
    for i in range(len(list_of_np_arrays)):
        size_1_array = list_of_np_arrays[i].shape[0]
        size_2_array = list_of_np_arrays[i].shape[1]
        numpy_arr[gap_horizontal:gap_horizontal+size_1_array,gap_vertical:gap_vertical+size_2_array] = list_of_np_arrays[i]
        gap_horizontal += size_1_array
        gap_vertical +=size_2_array
    return numpy_arr





def list_to_vec(list_of_np_arrays):
    size_2 = len(list_of_np_arrays)
    for i in range(size_2):
        if i==0:
            numpy_arr = list_of_np_arrays[i].reshape(-1,1)
        else:
            numpy_arr = np.r_[numpy_arr, list_of_np_arrays[i].reshape(-1,1)]
    return numpy_arr





def local_ecm(Matrixlist, vectorlist, swap_functions, constrain_sum_of_weights ):
    Number_Of_Clusters = len(Matrixlist)
    z_i = []
    w_i = []
    z = None
    for i in range(Number_Of_Clusters):
        z_i.append([])
        w_i.append([])
    unsuccesfull_index = 0
    if swap_functions ==True:
        Matrixlist = Matrixlist[::-1]
    for i in range(Number_Of_Clusters):
        hyper_reduction_element_selector = EmpiricalCubatureMethod()
        uu, ss, vv, ee = RandomizedSingularValueDecomposition().Calculate(Matrixlist[i].T)
        hyper_reduction_element_selector.SetUp( vv.T, Weights=vectorlist,InitialCandidatesSet = z, constrain_sum_of_weights=constrain_sum_of_weights)  #add again z
        hyper_reduction_element_selector.Initialize()
        hyper_reduction_element_selector.Calculate()
        if not hyper_reduction_element_selector.success:
            unsuccesfull_index+=1
            hyper_reduction_element_selector = EmpiricalCubatureMethod()
            hyper_reduction_element_selector.SetUp(vv.T, Weights=vectorlist, InitialCandidatesSet = None, constrain_sum_of_weights=constrain_sum_of_weights)
            hyper_reduction_element_selector.Initialize()
            hyper_reduction_element_selector.Calculate()
        w_i[i] = np.squeeze(hyper_reduction_element_selector.w)
        z_i[i] = np.squeeze(hyper_reduction_element_selector.z)
        if z is None:
            z = z_i[i]
        else:
            z = np.union1d(z,z_i[i])
    if swap_functions ==True:
        w_i = w_i[::-1]
        z_i = z_i[::-1]
    WeightsMatrix = np.zeros(( (Matrixlist[0].shape)[0],Number_Of_Clusters))
    for i in range(Number_Of_Clusters):
        for j in range(np.size(z_i[i])):
            try:
                WeightsMatrix[z_i[i][j] , i] = w_i[i][j]
            except:
                #single number found
                WeightsMatrix[z_i[i], i] = w_i[i]
    return z, WeightsMatrix





def independent_ecms(Matrixlist, vectorlist):
    Number_Of_Clusters = len(Matrixlist)
    z_i = []
    w_i = []
    for i in range(Number_Of_Clusters):
        hyper_reduction_element_selector = EmpiricalCubatureMethod()
        uu, ss, vv, ee = RandomizedSingularValueDecomposition().Calculate(Matrixlist[i].T)
        hyper_reduction_element_selector.SetUp(vv.T, Weights=vectorlist)
        hyper_reduction_element_selector.Initialize()
        hyper_reduction_element_selector.Calculate()
        w_i.append(np.squeeze(hyper_reduction_element_selector.w))
        z_i.append(np.squeeze(hyper_reduction_element_selector.z))
    return z_i, w_i





def global_ecm(GlobalMatrix, vectorlist):
    integrand = np.block(GlobalMatrix).T
    uu, ss, vv, ee = RandomizedSingularValueDecomposition().Calculate(integrand)
    hyper_reduction_element_selector = EmpiricalCubatureMethod()
    hyper_reduction_element_selector.SetUp( vv.T , Weights=vectorlist )
    hyper_reduction_element_selector.Initialize()
    hyper_reduction_element_selector.Calculate()
    return hyper_reduction_element_selector.z, hyper_reduction_element_selector.w





def function_1(X,power):
    return (X**(power)).reshape(-1,1)




def function_2(X,power):
    return np.c_[np.ones(X.shape), X**(power)]




def run_example(number_of_functions, number_of_candidate_Gauss_points, function_to_use, swap_functions, constrain_sum_of_weights):
    n = number_of_functions
    M = number_of_candidate_Gauss_points

    FunctionEvaluations = []
    BasisFunctionEvaluations = []
    GroundTruth = []

    [X,W] = gauss_quad(M) #Gauss points and weights as candidates


    for i in range(n):
        if function_to_use==1:
            Abar = function_1(X,i)
        else:
            Abar = function_2(X,i)
        FunctionEvaluations.append(Abar)
        AbarW = np.sqrt(W).reshape(-1,1)* Abar
        uu, ss, vv, ee = RandomizedSingularValueDecomposition().Calculate(AbarW)
        BasisFunctionEvaluations.append(uu*(1/np.sqrt(W.reshape(-1,1))))
        GroundTruth.append(BasisFunctionEvaluations[i].T@W)

    plot_original_function_1D(X,FunctionEvaluations)
    plot_basis_functions_1D(X,BasisFunctionEvaluations)


    A = list_to_mat(BasisFunctionEvaluations)
    b = list_to_vec(GroundTruth)
    c = np.ones(M*n)


    ### LP
    lp_matrices = []
    methods = ['highs' , 'highs-ds', 'highs-ipm', 'interior-point' , 'revised simplex' , 'simplex', '']
    for i in range(6):
        res = linprog(c, A_eq=A.T, b_eq=b, bounds=(0, None), method=methods[i] )
        #this makes the values very close to zero dissapear
        indexes_lp = np.where(res.x < np.zeros(res.x.shape)+1e-8)[0]
        res.x [indexes_lp]=0
        matrix = (res.x).reshape(n,M)
        lp_matrices.append(matrix)
    plot_lp_sparsity(lp_matrices,methods,'sparsity_linear_programming')
    indexes_lp = np.where(res.x > np.zeros(res.x.shape)+1e-6)
    weights_lp = res.x[indexes_lp]



    ### Local ECM
    indexes_local_ecm, weights_local_ecm = local_ecm(FunctionEvaluations, W, swap_functions, constrain_sum_of_weights)
    plot_sparsity(weights_local_ecm.T, 'local_ecm')
    weights_local_ecm = weights_local_ecm.T.reshape(-1)
    indexes_local_ecm = np.where(weights_local_ecm > np.zeros(weights_local_ecm.shape)+1e-6)
    weights_local_ecm = weights_local_ecm[indexes_local_ecm]




    ### Multiple indepenent ECMs
    index_list_independent_ecms, weight_list_independent_ecm  = independent_ecms(FunctionEvaluations, W)
    matrix = np.zeros(matrix.shape)
    for i in range(n):
        matrix[i, index_list_independent_ecms[i]] = 1
    plot_sparsity(matrix, 'multiple_independent')




    ### Solving a single globalECM problem
    index_single_ecm, weights_single_ecm  = global_ecm(FunctionEvaluations, W)
    matrix = np.zeros(matrix.shape)
    for i in range(len(index_single_ecm)):
        matrix[:, index_single_ecm[i]] = 1
    plot_sparsity(matrix, 'global')



    #Check LP approximation
    lp = A[indexes_lp].T@weights_lp



    #Check Local ECM approximation
    local_ecm_approximation = A[indexes_local_ecm].T@weights_local_ecm



    #Check Independent ECMs
    independent_ecm_approximation = []
    for i in range(n):
        if i ==0:
            independent_ecm_approximation = (BasisFunctionEvaluations[i].T[:,index_list_independent_ecms[i]]@ weight_list_independent_ecm[i].reshape(-1,1)).reshape(-1,1)
        else:
            independent_ecm_approximation = np.r_[independent_ecm_approximation, ( BasisFunctionEvaluations[i].T[:,index_list_independent_ecms[i]]@weight_list_independent_ecm[i].reshape(-1,1)).reshape(-1,1)  ]   #ppend(np.squeeze(A_i[i][index_list_independent_ecms[i]]*weight_list_independent_ecm[i]))



    #Check Global ECM
    integrand = np.block(BasisFunctionEvaluations).T
    global_ecm_approximation = integrand[:,index_single_ecm]@weights_single_ecm



    # print all approximations
    for i in range(n):
        print('exact:', b[i][0], 'local ecm:', local_ecm_approximation[i], '  lp: ', lp[i], 'independent ecms', independent_ecm_approximation[i],'  global ecm', global_ecm_approximation[i])


    #plotting all approximations
    plot_ground_truth([b, lp, global_ecm_approximation, independent_ecm_approximation,local_ecm_approximation], ['exact', 'lp','global ecm','independent ecms','local ecm'])





if __name__=='__main__':

    number_of_functions = 2
    number_of_candidate_Gauss_points = 20

    function_to_use = 1 # 1 or 2
    swap_functions = False # True or False. This changes the order of the functions to integrate
    constrain_sum_of_weights = True

    run_example(number_of_functions, number_of_candidate_Gauss_points, function_to_use, swap_functions, constrain_sum_of_weights)
