### Declaring functions for ECM

import numpy as np
from matplotlib import pyplot as plt
from empirical_cubature_method import EmpiricalCubatureMethod

### using LateX fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
###

def generate_G(rows=6, cols=2):
    np.random.seed(9)  #setting a seed to ensure consistent and nicely looking results
    a = np.random.rand(rows,cols)
    U,_,_  = np.linalg.svd(a, full_matrices=False)
    G = U
    return G


def generate_illposed_G(rows=6, cols=2):
    np.random.seed(9)  #setting a seed to ensure consistent and nicely looking results
    a = np.random.rand(rows,cols)
    b = np.sum(a, axis=0)/6
    a = a - b #making the sum of the rows to be zero
    U,_,_  = np.linalg.svd(a, full_matrices=False)
    G = U.T
    return G


def plot_ECM_Steps(G, W, Z):
    colours = ['yellow','cyan','green','cyan','black','magenta']
    r = []
    G_w = []
    r.append(np.sum(G, axis=1))
    green_lines = None
    magenta_lines = None
    for j in range(np.size(Z)):
        Gz = G[:,Z[j]]
        G_w.append(Gz*W[j])
        Gz = Gz.reshape(-1,1)
        r.append(r[j] - G_w[j])
        P = (Gz@Gz.T)/(Gz.T@Gz)
        if green_lines is None:
            p = P@r[j]
            projection = p
            green_lines = np.array([G_w[j][0], G_w[j][1]]).reshape(-1,1)
            green_lines =  np.c_[green_lines, np.array([r[0][0], r[0][1]]).reshape(-1,1)]
            magenta_lines = projection.reshape(-1,1)
            magenta_lines = np.c_[magenta_lines, np.array([r[0][0], r[0][1]]).reshape(-1,1)]
            rr = r[j]-p
        else:
            p = rr
            projection = np.c_[projection, rr]
            p = P@r[j]
            projection = np.c_[projection, p]
            green_lines =  np.c_[green_lines, np.array([G_w[j][0], G_w[j][1]]).reshape(-1,1)]
            green_lines =  np.c_[green_lines, np.array([r[0][0], r[0][1]]).reshape(-1,1)]
            magenta_lines = np.c_[magenta_lines, rr]
            magenta_lines = np.c_[magenta_lines, p]
            rr = r[j]-p

    #Step 1 plot the arrows
    ax = plt.axes()
    if True:
        #plotting the 2D vectors (2 modes were retrieved from the SVD of the residual matrix...
        # the number of arrows equals the number of elements)
        for i in range(G.shape[1]):
            ax.arrow(0.0, 0.0, G[0,i],G[1,i], head_width=np.max(G[:]*0.025))
            ax.text(G[0,i]+0.05,G[1,i]+0.05, str(i+1), ha='center', va='center')
    ax.axis('equal')
    ax.set_title('Step 1: get the candidate vectors', fontweight='bold')
    #plt.legend()
    plt.ylim([-1, 1.5])
    plt.xlim([-0.5, 2.5])
    plt.tight_layout()
    plt.savefig('Step1.pdf', bbox_inches='tight')
    plt.show()


    #Step 2 plot the original residual
    ax = plt.axes()
    if True:
        #plotting the 2D vectors (2 modes were retrieved from the SVD of the residual matrix...
        # the number of arrows equals the number of elements)
        for i in range(G.shape[1]):
            ax.arrow(0.0, 0.0, G[0,i],G[1,i], head_width=np.max(G[:]*0.025))
    ax.arrow(0.0, 0.0, r[0][0],r[0][1], color='red', head_width=np.max(G[:]*0.025), label=r'$||r_0||$  = '+  "{:.2f}".format(np.linalg.norm(r[0]) )  )
    #ax.axis('equal')
    ax.set_title('Step 2: sum all vectors to obtain the original residual',fontweight='bold')
    plt.legend()
    plt.ylim([-1, 1.5])
    plt.xlim([-0.5, 2.5])
    plt.tight_layout()
    plt.savefig('Step2.pdf', bbox_inches='tight')
    plt.show()


    #Step 3 find the projection onto the most positvely parallel vector
    ax = plt.axes()
    if True:
        #plotting the 2D vectors (2 modes were retrieved from the SVD of the residual matrix...
        # the number of arrows equals the number of elements)
        for i in range(G.shape[1]):
            ax.arrow(0.0, 0.0, G[0,i],G[1,i], head_width=np.max(G[:]*0.025))
    ax.arrow(0.0, 0.0, r[0][0],r[0][1], color='red', head_width=np.max(G[:]*0.025), label=r'$||r_0||$  = '+  "{:.2f}".format(np.linalg.norm(r[0]) )  )
    ax.arrow(0.0, 0.0, projection[0][0],projection[1][0], color='magenta', head_width=np.max(G[:]*0.025))
    ax.plot(magenta_lines[0,0:2], magenta_lines[1,0:2], 'm--')
    ax.set_title('Step 3: find the projection onto the most positvely parallel vector',fontweight='bold')
    #ax.axis('equal')
    #plt.legend()
    plt.ylim([-1, 1.5])
    plt.xlim([-0.5, 2.5])
    plt.tight_layout()
    plt.savefig('Step3.pdf', bbox_inches='tight')
    plt.show()


    #Step 4 update the residual
    ax = plt.axes()
    if True:
        #plotting the 2D vectors (2 modes were retrieved from the SVD of the residual matrix...
        # the number of arrows equals the number of elements)
        for i in range(G.shape[1]):
            ax.arrow(0.0, 0.0, G[0,i],G[1,i], head_width=np.max(G[:]*0.025))
    ax.arrow(0.0, 0.0, r[0][0],r[0][1], color='red', head_width=np.max(G[:]*0.025),  label=r'$||r_0||$  = '+  "{:.2f}".format(np.linalg.norm(r[0]) , alpha=0.3))
    ax.arrow(0.0, 0.0, projection[0][0],projection[1][0], color='magenta', head_width=np.max(G[:]*0.025), alpha=0.3)
    ax.arrow(0.0, 0.0, projection[0][1],projection[1][1], color='red', head_width=np.max(G[:]*0.025))
    ax.plot(magenta_lines[0,0:2], magenta_lines[1,0:2], 'm--')
    ax.axis('equal')
    ax.set_title('Step 4: update the residual',fontweight='bold')
    #plt.legend()
    plt.ylim([-1, 1.5])
    plt.xlim([-0.5, 2.5])
    plt.tight_layout()
    plt.savefig('Step4.pdf', bbox_inches='tight')
    plt.show()


    #Step 5 find vector most positely parallel to the updated residual
    ax = plt.axes()
    if True:
        #plotting the 2D vectors (2 modes were retrieved from the SVD of the residual matrix...
        # the number of arrows equals the number of elements)
        for i in range(G.shape[1]):
            ax.arrow(0.0, 0.0, G[0,i],G[1,i], head_width=np.max(G[:]*0.025))
    ax.arrow(0.0, 0.0, r[0][0],r[0][1], color='red', head_width=np.max(G[:]*0.025),  label=r'$||r_0||$  = '+  "{:.2f}".format(np.linalg.norm(r[0]) , alpha=0.3))
    ax.arrow(0.0, 0.0, projection[0][0],projection[1][0], color='magenta', head_width=np.max(G[:]*0.025), alpha=0.3)
    ax.arrow(0.0, 0.0, projection[0][1],projection[1][1], color='red', head_width=np.max(G[:]*0.025))
    for i in range(0, int(np.size(Z)*2), 2):
        ax.plot(magenta_lines[0,i:i+2], magenta_lines[1,i:i+2], 'm--')
    #ax.axis('equal')
    ax.set_title('Step 5: find vector most positely parallel to the updated residual',fontweight='bold')
    #plt.legend()
    plt.ylim([-1, 1.5])
    plt.xlim([-0.5, 2.5])
    plt.tight_layout()
    plt.savefig('Step5.pdf', bbox_inches='tight')
    plt.show()


    #Step 6 find weights for the two vectors found that minimize the original residual
    ax = plt.axes()
    if True:
        #plotting the 2D vectors (2 modes were retrieved from the SVD of the residual matrix...
        # the number of arrows equals the number of elements)
        for i in range(G.shape[1]):
            ax.arrow(0.0, 0.0, G[0,i],G[1,i], head_width=np.max(G[:]*0.025))
    ax.arrow(0.0, 0.0, r[0][0],r[0][1], color='red', head_width=np.max(G[:]*0.025), label=r'$||r_0||$  = '+  "{:.2f}".format(np.linalg.norm(r[0]) , alpha=0.3))
    ax.arrow(0.0, 0.0, r[-1][0],r[-1][1], color='red', head_width=np.max(G[:]*0.025),  label=r'$||r_{END}||$  = '+  "{:.2e}".format(np.linalg.norm(r[-1]) ))
    for i in range(len(r) -1):
        ax.arrow(0.0, 0.0, G_w[i][0],G_w[i][1], color=colours[i], head_width=np.max(G[:]*0.025), label='G[: , z_'+str(i)+'] @ w_'+str(i))
    for i in range(0, int(np.size(Z)*2), 2):
        ax.plot(green_lines[0,i:i+2], green_lines[1,i:i+2], 'g--')
    #ax.axis('equal')
    ax.set_title('Step 6: find weights for the vectors found that minimize the original residual',fontweight='bold')
    #plt.legend()
    ax.legend()
    plt.ylim([-1, 1.5])
    plt.xlim([-0.5, 2.5])
    plt.tight_layout()
    plt.savefig('Step6.pdf', bbox_inches='tight')
    plt.show()




def PlotIllPosedG(G):
    r = []
    r.append(np.sum(G, axis=1))

    #Step 2 plot the original residual
    ax = plt.axes()
    if True:
        #plotting the 2D vectors (2 modes were retrieved from the SVD of the residual matrix...
        # the number of arrows equals the number of elements)
        for i in range(G.shape[1]):
            ax.arrow(0.0, 0.0, G[0,i],G[1,i], head_width=np.max(G[:]*0.025))
    ax.arrow(0.0, 0.0, r[0][0],r[0][1], color='red', head_width=np.max(G[:]*0.025), label='r0 norm = '+str(np.linalg.norm(r[0])))
    #ax.axis('equal')
    ax.set_title('Step 2: sum all vectors to obtain the original residual',fontweight='bold')
    plt.legend()
    plt.ylim([-1, 1])
    plt.xlim([-1, 1])
    plt.show()





def launch_ecm_on_2D_matrix():

    G = generate_G()
    ECM = EmpiricalCubatureMethod() # Setting up Empirical Cubature Method problem
    Weights = np.ones(np.shape(G)[0]) # we arbitrary set W to the vector of ones
    ECM.SetUp(G.T, Weights = Weights, constrain_sum_of_weights=False)
    ECM.Run()
    W = np.squeeze(ECM.w)
    Z = ECM.z
    plot_ECM_Steps(G.T, W, Z)



if __name__=='__main__':
    launch_ecm_on_2D_matrix()