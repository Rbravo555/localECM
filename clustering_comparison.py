import numpy as np
np.float = float #deprecated float alias
from matplotlib import pyplot as plt

#import kmeans
from sklearn.cluster import KMeans

#fuzzy c-means stuff
import skfuzzy as fuzz
###########

#kmediods
from sklearn_extra.cluster import KMedoids
######

### using latex in plots
plt.rc('text', usetex=True)
plt.rc('font',family='serif')


def kmeans_test(test_data):
    n_clusters = 5
    kmeans_object = KMeans(n_clusters=n_clusters, random_state=0).fit(test_data)
    for j in range(n_clusters):
        plt.scatter(test_data[:, 0][kmeans_object.labels_==j], test_data[:, 1][kmeans_object.labels_==j])
    centroids_to_plot = (kmeans_object.cluster_centers_).T
    plt.scatter(centroids_to_plot[0,:], centroids_to_plot[1,:], c='k',marker="s", s= 150)
    plt.title(r"\textbf{k-means clustering}")
    plt.savefig('k-means clustering.pdf',bbox_inches='tight' )
    plt.show()


def fuzzy_c_means(test_data):
    ncenters = 5
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(test_data.T, ncenters, 2, error=0.005, maxiter=1000, init=None)
    #setting cluster memebership to the largest percentage of belonging
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        plt.scatter(test_data[:, 0],test_data[:, 1], s=150*u[j,:])
    # Mark the center of each fuzzy cluster
    for pt in cntr:
        plt.scatter(pt[0], pt[1],c='k',marker="s",s= 150)
    plt.title(r"\textbf{fuzzy c-means clustering}")
    plt.savefig('fuzzy c-means clustering.pdf',bbox_inches='tight' )
    plt.show()

def kmediods_test(test_data):
    n_clusters = 5
    kmedoids_object = KMedoids(n_clusters=n_clusters,random_state=0).fit(test_data)
    for j in range(n_clusters):
        plt.scatter(test_data[:, 0][kmedoids_object.labels_==j], test_data[:, 1][kmedoids_object.labels_==j])
    centroids_to_plot = (kmedoids_object.cluster_centers_).T
    plt.scatter(centroids_to_plot[0,:], centroids_to_plot[1,:], c='k',marker="s", s=150)
    plt.title(r"\textbf{k-medoids clustering}")
    plt.savefig('k-medoids clustering.pdf',bbox_inches='tight' )
    plt.show()


def E_p(u, c):
    """
    c: direction vector onto which to project
    u: vector or colection of column vectors to project onto the direction of c
    """
    c = c.reshape(-1,1)
    if len(u.shape)==1:
        u = u.reshape(-1,1)
    projection_of_u_onto_c = ((c@c.T) / (c.T@c)) @ u
    projection_error = np.linalg.norm(u - projection_of_u_onto_c, axis=0) / np.linalg.norm(u,axis=0)
    return projection_error

def PEBL(Snapshots, bisection_tolerance=0.15,  POD_tolerance=1e-3):
    #stage 1, generation of bisection tree with accuracy 'bisection_tolerance'
    max_index = np.argmax( np.linalg.norm(Snapshots, axis=0) )
    first_snapshot = Snapshots[:,max_index]
    Tree = Node([first_snapshot, Snapshots])
    bisect_flag = True
    while bisect_flag == True:
        bisect_flag = False
        for leaf in Tree.leaves():
            errors = E_p(leaf.val[1], leaf.val[0])
            max_error = max(errors)
            print(max_error)
            if max_error > bisection_tolerance:
                bisect_flag = True
                #find next anchor point
                max_index = np.argmax(errors)
                c_new = leaf.val[1][:,max_index]
                new_errors = E_p(leaf.val[1], c_new)
                indexes_left = np.where( errors <= new_errors)
                indexes_right = np.where( errors > new_errors)
                #divide the snapshots among the two children
                leaf.left =  Node([leaf.val[0], leaf.val[1][:,indexes_left[0]]])
                leaf.right = Node([c_new, leaf.val[1][:,indexes_right[0]]])
                leaf.val[1] = None
    #stage 2, generation of local POD bases'
    for leaf in Tree.leaves():
        Phi_i = ObtainBasis(leaf.val[1], POD_tolerance)
        leaf.val.append(Phi_i)
    return Tree


def ObtainBasis(Snapshots, truncation_tolerance=0):
        U,_,_= truncated_svd(Snapshots,truncation_tolerance)
        return U

def truncated_svd(Matrix, epsilon=0):
    M,N=np.shape(Matrix)
    dimMATRIX = max(M,N)
    U, s, V = np.linalg.svd(Matrix, full_matrices=False) #U --> M xN, V --> N x N
    V = V.T
    tol = dimMATRIX*np.finfo(float).eps*max(s)/2
    R = np.sum(s > tol)  # Definition of numerical rank
    if epsilon == 0:
        K = R
    else:
        SingVsq = np.multiply(s,s)
        SingVsq.sort()
        normEf2 = np.sqrt(np.cumsum(SingVsq))
        epsilon = epsilon*normEf2[-1] #relative tolerance
        T = (sum(normEf2<epsilon))
        K = len(s)-T
    K = min(R,K)
    return U[:, :K], s[:K], V[:, :K]


class Node:
    def __init__(self, val):
        self.left = None
        self.right = None
        self.val = val

    def leaves(self):
        current_nodes = [self]
        leaves = []

        while len(current_nodes) > 0:
            next_nodes = []
            for node in current_nodes:
                if node.left is None and node.right is None:
                    leaves.append(node)
                    continue
                if node.left is not None:
                    next_nodes.append(node.left)
                if node.right is not None:
                    next_nodes.append(node.right)
            current_nodes = next_nodes
        return leaves

def pebl_test(test_data):
    Tree = PEBL(test_data.T, 0.68)
    plt.figure()
    for leaf in Tree.leaves():
        plt.scatter(leaf.val[1][0,:], leaf.val[1][1,:])
        plt.scatter(leaf.val[0][0], leaf.val[0][1], c='k',marker="s", s=150)
    plt.title(r"\textbf{PEBL clustering}")
    plt.savefig('PEBL clustering.pdf',bbox_inches='tight' )
    plt.show()




if __name__=='__main__':
    np.random.seed(42) #seed the random sampling

    #generate seed random data
    test_data = np.random.rand(1000,2)
    test_data -= test_data.mean(axis = 0) #centering

    #launch tests
    kmeans_test(test_data)
    kmediods_test(test_data)
    fuzzy_c_means(test_data)
    pebl_test(test_data)
