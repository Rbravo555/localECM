
import numpy as np
from scipy.special.orthogonal import p_roots
from scipy.optimize import linprog
from empirical_cubature_method import EmpiricalCubatureMethod
from randomized_singular_value_decomposition import RandomizedSingularValueDecomposition
from plot_scripts import *




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




def local_ecm(Matrixlist, W, constrain_sum_of_weights , use_L2_weighting ):
    Number_Of_Clusters = len(Matrixlist)
    z_i = []
    w_i = []
    z = None
    for i in range(Number_Of_Clusters):
        z_i.append([])
        w_i.append([])
    unsuccesfull_index = 0
    for i in range(Number_Of_Clusters):
        hyper_reduction_element_selector = EmpiricalCubatureMethod()

        if use_L2_weighting:
            Abar = np.sqrt(W).reshape(-1,1) * Matrixlist[i]
        else:
            Abar = Matrixlist[i]
        U_bar, ss, vv, ee = RandomizedSingularValueDecomposition().Calculate(Abar)
        hyper_reduction_element_selector.SetUp( U_bar.T, Weights=W,InitialCandidatesSet = z, constrain_sum_of_weights=constrain_sum_of_weights, use_L2_weighting=use_L2_weighting)
        hyper_reduction_element_selector.Run()
        if not hyper_reduction_element_selector.success:
            unsuccesfull_index+=1
            hyper_reduction_element_selector.SetUp(U_bar.T, Weights=W, InitialCandidatesSet = None, constrain_sum_of_weights=constrain_sum_of_weights, use_L2_weighting=use_L2_weighting)
            hyper_reduction_element_selector.Run()
        w_i[i] = np.squeeze(hyper_reduction_element_selector.w)
        z_i[i] = np.squeeze(hyper_reduction_element_selector.z)

        print('the sum of the weights is: ', np.sum(np.squeeze(hyper_reduction_element_selector.w)))

        if z is None:
            z = z_i[i]
        else:
            z = np.union1d(z,z_i[i])

    WeightsMatrix = np.zeros(( (Matrixlist[0].shape)[0],Number_Of_Clusters))
    for i in range(Number_Of_Clusters):
        for j in range(np.size(z_i[i])):
            try:
                WeightsMatrix[z_i[i][j] , i] = w_i[i][j]
            except:
                #single number found
                WeightsMatrix[z_i[i], i] = w_i[i]

    return z, WeightsMatrix





def independent_ecms(Matrixlist, W, constrain_sum_of_weights, use_L2_weighting):
    Number_Of_Clusters = len(Matrixlist)
    Number_Of_Gauss_Points = np.size(W)
    WeightsMatrix = np.zeros((Number_Of_Clusters,Number_Of_Gauss_Points))

    for i in range(Number_Of_Clusters):
        hyper_reduction_element_selector = EmpiricalCubatureMethod()
        if use_L2_weighting:
            Abar = np.sqrt(W).reshape(-1,1) * Matrixlist[i]
        else:
            Abar = Matrixlist[i]
        U_bar, ss, vv, ee = RandomizedSingularValueDecomposition().Calculate(Abar)
        hyper_reduction_element_selector.SetUp(U_bar.T, Weights=W, constrain_sum_of_weights=constrain_sum_of_weights, use_L2_weighting=use_L2_weighting)
        hyper_reduction_element_selector.Run()
        print('sum of weights = ', np.sum(hyper_reduction_element_selector.w))
        WeightsMatrix[i, hyper_reduction_element_selector.z] = (hyper_reduction_element_selector.w).flatten()

    return WeightsMatrix





def global_ecm(GlobalMatrix, W,constrain_sum_of_weights, use_L2_weighting):

    integrand = np.block(GlobalMatrix)
    if use_L2_weighting:
        Abar = np.sqrt(W).reshape(-1,1) * integrand
    else:
        Abar = integrand
    U_bar, ss, vv, ee = RandomizedSingularValueDecomposition().Calculate(Abar)
    hyper_reduction_element_selector = EmpiricalCubatureMethod()
    hyper_reduction_element_selector.SetUp( U_bar.T , Weights=W, constrain_sum_of_weights=constrain_sum_of_weights, use_L2_weighting=use_L2_weighting)
    hyper_reduction_element_selector.Run()


    Number_Of_Clusters = len(GlobalMatrix)
    Number_Of_Gauss_Points = np.size(W)
    WeightsMatrix = np.zeros((Number_Of_Clusters,Number_Of_Gauss_Points))
    WeightsMatrix[:, hyper_reduction_element_selector.z] = (hyper_reduction_element_selector.w).flatten()

    return WeightsMatrix





def GetSparsestSolution(list_of_weights, methods=[None]):

    sparsest_index = 0
    sparsest_solution = np.shape(list_of_weights[sparsest_index])[1]

    for i in range(len(list_of_weights)):
        sparsity = np.linalg.norm( np.sum(list_of_weights[i], axis = 0) , 0 )
        if  sparsity < sparsest_solution :
            sparsest_solution = sparsity
            sparsest_index = i


    return sparsest_solution, methods[sparsest_index], sparsest_index








def function_1(X,power):
    return (X**(power)).reshape(-1,1)




def function_2(X,power):
    return np.c_[np.ones(X.shape), X**(power)]




def run_example(number_of_functions, number_of_candidate_Gauss_points, function_to_use, constrain_sum_of_weights, use_L2_weighting):
    n = number_of_functions
    M = number_of_candidate_Gauss_points

    FunctionEvaluations = []
    BasisFunctionEvaluations = []
    GroundTruth = []
    ExactIntegral = []
    PointsWithNonZeroWeights = {}

    [X,W] = gauss_quad(M) #Gauss points and weights as candidates


    for i in range(n):
        if function_to_use==1:
            Abar = function_1(X,i)
        else:
            Abar = function_2(X,i)
        FunctionEvaluations.append(Abar)
        Abar = np.sqrt(W).reshape(-1,1)* Abar
        uu, ss, vv, ee = RandomizedSingularValueDecomposition().Calculate(Abar)
        BasisFunctionEvaluations.append(uu*(1/np.sqrt(W.reshape(-1,1))))
        GroundTruth.append(BasisFunctionEvaluations[i].T@W)
        ExactIntegral.append(FunctionEvaluations[i].T@W)

    if function_to_use==1:
        function_outputs = 1
    else:
        function_outputs = 2

    for component in range(function_outputs):
        plot_original_function_1D(X,FunctionEvaluations,component)
        plot_basis_functions_1D(X,BasisFunctionEvaluations,component)


    A = list_to_mat(FunctionEvaluations)
    b = list_to_vec(ExactIntegral)
    U_tilde = list_to_mat(BasisFunctionEvaluations)
    d_tilde = list_to_vec(GroundTruth)
    c = np.ones(M*n)



    ### Local ECM
    indexes_local_ecm, weights_local_ecm = local_ecm(FunctionEvaluations, W, constrain_sum_of_weights, use_L2_weighting)
    PointsWithNonZeroWeights["LocalECM"], _, _ = GetSparsestSolution([weights_local_ecm.T])
    plot_sparsity(weights_local_ecm.T, 'local_ecm')
    weights_local_ecm = weights_local_ecm.T.reshape(-1)
    indexes_local_ecm = np.where(weights_local_ecm > np.zeros(weights_local_ecm.shape)+1e-8)
    weights_local_ecm = weights_local_ecm[indexes_local_ecm]




    ### GlobalECM
    global_ecm_weights_matrix = global_ecm(FunctionEvaluations, W, constrain_sum_of_weights, use_L2_weighting)
    PointsWithNonZeroWeights["GlobalECM"], _, _ = GetSparsestSolution([global_ecm_weights_matrix])
    plot_sparsity(global_ecm_weights_matrix, 'global')
    global_ecm_weights_vector = global_ecm_weights_matrix.flatten()
    indexes_global_ecm = np.where( global_ecm_weights_vector > 0)[0]
    weights_global_ecm = global_ecm_weights_vector[indexes_global_ecm]





    ### Multiple indepenent ECMs
    independent_ecms_weights_matrix = independent_ecms(FunctionEvaluations, W, constrain_sum_of_weights, use_L2_weighting)
    PointsWithNonZeroWeights["IndependentECMs"], _, _ = GetSparsestSolution([independent_ecms_weights_matrix])
    plot_sparsity(independent_ecms_weights_matrix, 'multiple_independent')
    independent_ecms_weights_vector = independent_ecms_weights_matrix.flatten()
    indexes_independent_ecms = np.where( independent_ecms_weights_vector > 0)[0]
    weights_independent_ecms = independent_ecms_weights_vector[indexes_independent_ecms]





    ### LP
    lp_matrices = []
    methods = ['highs' , 'highs-ds', 'highs-ipm', 'interior-point' , 'revised simplex' , 'simplex']
    for i in range(6):
        res = linprog(c, A_eq=U_tilde.T, b_eq=d_tilde, bounds=(0, None), method=methods[i] )
        indexes_lp = np.where(res.x < np.zeros(res.x.shape)+1e-8)[0] #this makes the values very close to zero dissapear
        res.x [indexes_lp]=0
        matrix = (res.x).reshape(n,M)
        lp_matrices.append(matrix)
    plot_lp_sparsity(lp_matrices,methods,'sparsity_linear_programming')
    number_points_selected_lp, sparsest_lp_name, sparsest_lp_index = GetSparsestSolution(lp_matrices, methods)
    PointsWithNonZeroWeights["LP "+sparsest_lp_name] = number_points_selected_lp
    lp_solution = lp_matrices[sparsest_lp_index].flatten()
    indexes_lp = np.where( lp_solution > 0)[0]
    weights_lp = lp_solution[indexes_lp]




    #Check LP approximation
    lp = U_tilde[indexes_lp].T@weights_lp
    lp_exact_integral = A[indexes_lp].T@weights_lp

    #Check Local ECM approximation
    local_ecm_approximation = U_tilde[indexes_local_ecm].T@weights_local_ecm
    local_ecm_exact_integral = A[indexes_local_ecm].T@weights_local_ecm

    # #Check Independent ECMs
    independent_ecm_approximation = U_tilde[indexes_independent_ecms].T@weights_independent_ecms
    independent_ecm_exact_integral = A[indexes_independent_ecms].T@weights_independent_ecms


    #Check Global ECM
    global_ecm_approximation = U_tilde[indexes_global_ecm].T@weights_global_ecm
    global_ecm_exact_integral = A[indexes_global_ecm].T@weights_global_ecm


    #plotting number of selected points
    plot_selected_points_histogram(PointsWithNonZeroWeights)

    #ground truth U_tilde.T@W
    saving_title = 'ground_truth_approximation'
    data = [d_tilde, lp, global_ecm_approximation, independent_ecm_approximation,local_ecm_approximation]
    labels = ['ground truth', 'lp '+sparsest_lp_name,'global ecm','independent ecms','local ecm']
    ylabel = r'$\int u^{(i)} dx$'
    xlabel = r'$\tilde{d}_j$'
    plot_approximations(data, labels, ylabel, xlabel, saving_title)

    #error on ground truth approximation
    saving_title = 'ground_truth_error'
    d_tilde = d_tilde.flatten()
    data = [lp-d_tilde, global_ecm_approximation-d_tilde, independent_ecm_approximation-d_tilde,local_ecm_approximation-d_tilde]
    labels = ['lp '+sparsest_lp_name,'global ecm','independent ecms','local ecm']
    ylabel = r'error'
    xlabel = r'$\tilde{d}_j$'
    plot_approximations(data, labels, ylabel, xlabel, saving_title)

    #exact integral approximation A.T@W
    saving_title = 'exact_integral_approximation'
    data = [b, lp_exact_integral, global_ecm_exact_integral, independent_ecm_exact_integral,local_ecm_exact_integral]
    labels = ['exact', 'lp '+sparsest_lp_name,'global ecm','independent ecms','local ecm']
    ylabel = r'$\int a^{(i)} dx$'
    xlabel = r'$\tilde{b}_j$'
    plot_approximations(data, labels, ylabel, xlabel, saving_title)

    #error on exact integral approximation
    b = b.flatten()
    saving_title = 'exact_integral_error'
    data = [lp_exact_integral-b, global_ecm_exact_integral-b, independent_ecm_exact_integral-b,local_ecm_exact_integral-b]
    labels = ['lp '+sparsest_lp_name,'global ecm','independent ecms','local ecm']
    ylabel = r'error'
    xlabel = r'$\tilde{b}_j$'
    plot_approximations(data, labels, ylabel, xlabel, saving_title)





if __name__=='__main__':

    number_of_functions = 20
    number_of_candidate_Gauss_points = 50

    function_to_use = 2 # 1 or 2
    constrain_sum_of_weights = False #this avoids the trivial solution
    use_L2_weighting = True # True  # if True: d = G@\sqrt{W}; elif False: d = G@W

    run_example(number_of_functions, number_of_candidate_Gauss_points, function_to_use, constrain_sum_of_weights, use_L2_weighting)