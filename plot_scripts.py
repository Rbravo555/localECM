import numpy as np
from matplotlib import pyplot as plt
try:
    kkk
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    latex_available = True
except:
    latex_available = False

import itertools

# def plot_original_function_1D(X,A_i):
#     number_of_polynomials = len(A_i)

#     colours = ['k','m','g','b','c','r','y']
#     markers=['X','<','^','*','v','P','>','d','8','H']

#     counter = 0
#     for i in range(number_of_polynomials):
#         plt.plot(X, A_i[i], colours[counter], marker=markers[counter], label=r"$a^{{{}}}(x,{{{}}})$".format(i+1,i))
#         #plt.plot(X, A_i[i], label=r"$a^{{{}}}(x,{{{}}})$".format(i+1,i))
#         counter+=1
#     plt.xlabel(r'$x$', size=15, fontweight='bold')
#     plt.ylabel(r'$a^{(i+1)}$',size=15,fontweight='bold')
#     plt.legend()
#     plt.grid()
#     plt.savefig(f'1d_functions.pdf')
#     plt.show()



def plot_ground_truth(solutions, labels):

    colors = ['k', 'm', 'g', 'b', 'c', 'r', 'y']
    markers = ['X', '<', '^', '*', 'v', 'P', '>', 'd', '8', 'H']
    line_styles = ['-', '--', '-.', ':']

    color_cycle = itertools.cycle(colors)
    marker_cycle = itertools.cycle(markers)
    line_style_cycle = itertools.cycle(line_styles)

    number_of_solutions = len(solutions)

    for i in range(number_of_solutions):
        color = next(color_cycle)
        marker = next(marker_cycle)
        line_style = next(line_style_cycle)
        plt.plot( solutions[i] , color=color, marker=marker, linestyle=line_style,
                 label=labels[i])

    plt.xlabel(r'$x$', size=15, fontweight='bold')
    plt.ylabel(r'$\int a^{(i+1) dx }$', size=15, fontweight='bold')

    plt.xticks(rotation='vertical')  # Rotate x-axis tick labels
    plt.rcParams['font.size'] = 8  # Adjust the font size
    # Place the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.legend()
    plt.grid()
    plt.savefig(f'ground_truth.pdf',bbox_inches='tight')
    plt.show()











def plot_original_function_1D(X, A_i, componenet_to_plot=0):
    number_of_polynomials = len(A_i)

    colors = ['k', 'm', 'g', 'b', 'c', 'r', 'y']
    markers = ['X', '<', '^', '*', 'v', 'P', '>', 'd', '8', 'H']
    line_styles = ['-', '--', '-.', ':']

    color_cycle = itertools.cycle(colors)
    marker_cycle = itertools.cycle(markers)
    line_style_cycle = itertools.cycle(line_styles)

    for i in range(number_of_polynomials):
        color = next(color_cycle)
        marker = next(marker_cycle)
        line_style = next(line_style_cycle)
        Y = A_i[i][:,componenet_to_plot]
        plt.plot(X, Y , color=color, marker=marker, linestyle=line_style,
                 label=r"$a^{{{}}}(x,{{{}}})$".format(i + 1, i))

    plt.xlabel(r'$x$', size=15, fontweight='bold')
    plt.ylabel(r'$a^{(i+1)}$', size=15, fontweight='bold')

    plt.xticks(rotation='vertical')  # Rotate x-axis tick labels
    plt.rcParams['font.size'] = 8  # Adjust the font size
    # Place the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    #plt.legend()
    plt.grid()
    plt.savefig(f'1d_functions_{componenet_to_plot}.pdf',bbox_inches='tight')
    plt.show()








def plot_basis_functions_1D(X, A_i, componenet_to_plot=0):
    number_of_polynomials = len(A_i)

    colors = ['k', 'm', 'g', 'b', 'c', 'r', 'y']
    markers = ['X', '<', '^', '*', 'v', 'P', '>', 'd', '8', 'H']
    line_styles = ['-', '--', '-.', ':']

    color_cycle = itertools.cycle(colors)
    marker_cycle = itertools.cycle(markers)
    line_style_cycle = itertools.cycle(line_styles)

    for i in range(number_of_polynomials):
        color = next(color_cycle)
        marker = next(marker_cycle)
        line_style = next(line_style_cycle)
        try:
            if np.shape(A_i[i])[1]>=1:
                Y = A_i[i][:,componenet_to_plot]
        except:
            Y = A_i[i]
        plt.plot(X, Y, color=color, marker=marker, linestyle=line_style,
                 label=r"$u^{{{}}}(x)$".format(i + 1, i))

    plt.xlabel(r'$x$', size=15, fontweight='bold')
    plt.ylabel(r'$u^{(i+1)}$', size=15, fontweight='bold')

    plt.xticks(rotation='vertical')  # Rotate x-axis tick labels
    plt.rcParams['font.size'] = 8  # Adjust the font size
    # Place the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    #plt.legend()
    plt.grid()
    plt.savefig(f'1d_basis_functions_{componenet_to_plot}.pdf',bbox_inches='tight')
    plt.show()


def plot_lp_sparsity(lp_matrices,methods, title ):
    for i in range(len(lp_matrices)):
        # Plot the sparse representation (spy) of the matrix
        plt.subplot(2, 3, i+1)
        plt.spy(lp_matrices[i], markersize=2)
        plt.title(fr'{methods[i]}')
    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.savefig(f'{title}.pdf', bbox_inches='tight')
    plt.show()


# def plot_basis_functions_1D(X,A_i):
#     number_of_polynomials = len(A_i)

#     colours = ['k','m','g','b','c','r','y']
#     markers=['X','<','^','*','v','P','>','d','8','H']

#     counter = 0
#     for i in range(number_of_polynomials):
#         plt.plot(X, A_i[i], colours[counter], marker=markers[counter], label=r"$u^{{{}}}(x)$".format(i+1,i))
#         #plt.plot(X, A_i[i],label=r"$u^{{{}}}(x)$".format(i+1,i))
#         counter+=1
#     plt.xlabel(r'$x$', size=15, fontweight='bold')
#     plt.ylabel(r'$u^{(i+1)}$',size=15,fontweight='bold')
#     plt.legend()
#     plt.grid()
#     plt.savefig(f'1d_basis_functions.pdf')
#     plt.show()




def plot_spartity_plot(short_and_fat_sparse_matrix, title):
    plt.spy(short_and_fat_sparse_matrix)
    plt.xlabel(r'$\bar{\textbf{x}}_j$', size=15, fontweight='bold')
    plt.ylabel(r'$i$-th function',size=15,fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'sparsity_{title}.pdf', bbox_inches='tight')
    plt.show()


def plot_elements(tol_svd,to_svd_res,resuts_path):

    for svd in tol_svd:
        print('\n')
        for res in to_svd_res:
            nelems = np.load(resuts_path+f'Elementsvector_{svd}_{res}.npy')
            print('svd tol: ',svd, ' res_tol: ',res, ' number of elements: ', np.size(nelems))
        print('\n')



def plot_hysteresis(tol_svd,to_svd_res,resuts_path):
    markers=['s','^','+','H','*']
    counter = 0
    for svd in tol_svd:
        vy = np.load(resuts_path+'Velocity_y.npy')
        w = np.load(resuts_path+'narrowing.npy')
        plt.plot(vy, w, 'b', label = 'FOM', linewidth = 1.5 ) #linewidth = 3, alpha=0.5
        vy_rom = np.load(resuts_path+f'y_velocity_ROM_{svd}.npy')
        w_rom = np.load(resuts_path+f'narrowing_ROM_{svd}.npy')
        plt.plot(vy_rom, w_rom, label = f"ROM\_{svd}", linewidth = 1.5) #alpha=0.5
        for res in to_svd_res:
            vy_rom = np.load(resuts_path+f'y_velocity_HROM_{svd}_{res}.npy')
            w_rom = np.load(resuts_path+f'narrowing_HROM_{svd}_{res}.npy')
            plt.plot(vy_rom, w_rom, markevery=50,label = f"HROM\_{svd}\_{res}", linewidth = 1.5) #alpha=0.5 marker=markers[counter],
            counter+=1
        plt.xlabel(r'$v_y^*$', size=15, fontweight='bold')
        plt.ylabel(r'$w_c$',size=15,fontweight='bold')
        plt.legend()
        plt.grid()
        plt.show()

    #plt.title(r'FOM vs ROM', fontsize=20, fontweight='bold')
    # plt.xticks(np.arange(-3.5,0.25,0.25))  #TODO the ticks are not easily beautyfied, do it later :)
    # plt.yticks(np.arange(0,3.1,0.25))
    #plt.savefig('fom_vs_rom')


def plot_hysteresis_fom_rom(tol_svd,to_svd_res,resuts_path, trajectory):
    if trajectory == "train":
        add_to_name = ''
    elif trajectory == "test":
        add_to_name = '_test'

    colours = ['k','m','g','b','c','r','y']
    markers=['X','<','^','*','v','P','>','d','8','H']
    vy = np.load(resuts_path+f'Velocity_y{add_to_name}.npy')
    w = np.load(resuts_path+f'narrowing{add_to_name}.npy')
    plt.plot(vy, w, colours[0], marker=markers[0], markevery=50, label = 'FOM') #linewidth = 3, alpha=0.5
    #plt.plot(vy, w, colours[0], linewidth = 5, alpha=0.5)
    counter=1
    for svd in tol_svd:
        vy_rom = np.load(resuts_path+f'y_velocity_ROM_{svd}{add_to_name}.npy')
        w_rom = np.load(resuts_path+f'narrowing_ROM_{svd}{add_to_name}.npy')
        plt.plot(vy_rom, w_rom, colours[counter], marker=markers[counter], markevery=80+counter*15,label = r"$ROM \ \epsilon_{SOL}=$" + "{:.0e}".format(svd), alpha=0.9) #alpha=0.5, #alpha=0.5
        counter+=1
    plt.xlabel(r'$v_y^*$', size=15)
    plt.ylabel(r'$w_c$',size=15)
    plt.grid()
    #plt.xlim([-4,0])
    plt.legend()
    plt.savefig(f'hysteresis_fom_vs_rom_{trajectory}.pdf')
    plt.show()


    #plt.title(r'FOM vs ROM', fontsize=20, fontweight='bold')
    # plt.xticks(np.arange(-3.5,0.25,0.25))  #TODO the ticks are not easily beautyfied, do it later :)
    # plt.yticks(np.arange(0,3.1,0.25))
    #plt.savefig('fom_vs_rom')



def plot_hysteresis_rom_hrom(tol_svd,to_svd_res,resuts_path, trajectory):

    if trajectory == "train":
        add_to_name = ''
    elif trajectory == "test":
        add_to_name = '_test'

    colours = ['k','m','g','b','c','r','y']
    markers=['X','<','^','*','v','P','>','d','8','H']

    for svd in tol_svd:
        vy_rom = np.load(resuts_path+f'y_velocity_ROM_{svd}{add_to_name}.npy')
        w_rom = np.load(resuts_path+f'narrowing_ROM_{svd}{add_to_name}.npy')
        counter=0
        plt.plot(vy_rom, w_rom, colours[counter], marker=markers[counter], markevery=50,label = r"$ROM \ \epsilon_{SOL}=$" + "{:.0e}".format(svd), alpha=0.9) #alpha=0.5, #alpha=0.5
        for res in to_svd_res:
            vy_rom = np.load(resuts_path+f'y_velocity_HROM{add_to_name}_{svd}_{res}.npy')
            w_rom = np.load(resuts_path+f'narrowing_HROM{add_to_name}_{svd}_{res}.npy')
            plt.plot(vy_rom, w_rom, markevery=80+counter*15,
            label = r"$HROM \ \epsilon_{SOL}=$ " + "{:.0e}".format(svd) + r" ; $\epsilon_{RES}=$ "+"{:.0e}".format(res),
            linewidth = 1.5, marker=markers[counter]) #alpha=0.5 marker=markers[counter]
            counter+=1
        counter+=1
        plt.xlabel(r'$v_y^*$', size=15)
        plt.ylabel(r'$w_c$',size=15)
        plt.xlim([-4,1])
        plt.grid()
        plt.legend()
        plt.savefig(f'hysteresis_rom_vs_hrom_{svd}_{trajectory}.pdf')
        plt.show()


def GetPercentualError(reference, approx):
    return np.linalg.norm(reference - approx) / np.linalg.norm(reference) *100


def plot_errors_FOM_ROM(tol_svd,to_svd_res,resuts_path,trajectory):

    if trajectory == "train":
        add_to_name = ''
    elif trajectory == "test":
        add_to_name = '_test'

    FOM = np.load(resuts_path+f'SnapshotMatrix{add_to_name}.npy')
    for svd in tol_svd:
        ROM = np.load(resuts_path+f'ROM_snapshots{add_to_name}_{svd}.npy')
        print('FOM vs ROM error: ', GetPercentualError(FOM, ROM),'%\n')


def plot_errors_ROM_HROM(tol_svd,to_svd_res,resuts_path,trajectory):

    if trajectory == "train":
        add_to_name = ''
    elif trajectory == "test":
        add_to_name = '_test'

    for svd in tol_svd:
        ROM = np.load(resuts_path+f'ROM_snapshots{add_to_name}_{svd}.npy')
        print('\n solution_svd_truncation:',svd)
        for res in to_svd_res:
            print('residual_svd_truncation:',res)
            HROM = np.load(resuts_path+f'HROM_snapshots{add_to_name}_{svd}_{res}.npy')
            print('ROM vs HROM error: ', GetPercentualError(ROM,HROM),'%')
        print('\n')






def error_QoI_FOM_vs_ROM(tol_svd,to_svd_res,resuts_path,trajectory):

    if trajectory == "train":
        add_to_name = ''
    elif trajectory == "test":
        add_to_name = '_test'



    FOM = np.load(resuts_path+f'Velocity_y{add_to_name}.npy')
    for svd in tol_svd:
        ROM = np.load(resuts_path+f'y_velocity_ROM{add_to_name}_{svd}.npy')
        print('FOM vs ROM error: ', GetPercentualError(FOM, ROM),'%\n')





def error_QoI_ROM_vs_HROM(tol_svd,to_svd_res,resuts_path,trajectory):

    if trajectory == "train":
        add_to_name = ''
    elif trajectory == "test":
        add_to_name = '_test'



    for svd in tol_svd:
        ROM = np.load(resuts_path+f'y_velocity_ROM{add_to_name}_{svd}.npy')
        print('\nsolution_svd_truncation:',svd)
        for res in to_svd_res:
            #print('residual_svd_truncation:',res)
            HROM = np.load(resuts_path+f'y_velocity_HROM{add_to_name}_{svd}_{res}.npy')
            print('ROM vs HROM error: ', GetPercentualError(ROM,HROM),'%')
        print('\n')




def plot_number_of_modes(tol_svd,results_path_rom):
    for svd in tol_svd:
        number_of_modes = np.load(results_path_rom+f'{svd}.npy').shape[1]
        print(f'\nfor a tolerance of {svd}, {number_of_modes} were selected')
        print('\n')





if __name__=='__main__':

    resuts_path_hrom = './HROM/'
    results_path_rom = './ROM/'
    resuts_path = './Results/' #'./Results_slightly_shifted/'
    trajectory = "train" # test or train

    tol_svd = [1e-3,1e-4,1e-5,1e-6]
    to_svd_res = [1e-3,1e-4,1e-5,1e-6]

    #TODO add testing trajectory plots!!!

    # plot_hysteresis(tol_svd,to_svd_res,resuts_path)
    # plot_hysteresis_fom_rom(tol_svd,to_svd_res,resuts_path, trajectory)
    # plot_hysteresis_rom_hrom(tol_svd,to_svd_res,resuts_path, trajectory)
    # plot_number_of_modes(tol_svd,results_path_rom)
    # plot_elements(tol_svd,to_svd_res,resuts_path_hrom )
    plot_errors_FOM_ROM(tol_svd,to_svd_res,resuts_path, trajectory)
    #plot_errors_ROM_HROM(tol_svd,to_svd_res,resuts_path, trajectory)
    #error_QoI_FOM_vs_ROM(tol_svd,to_svd_res,resuts_path, trajectory)
    #error_QoI_ROM_vs_HROM(tol_svd,to_svd_res,resuts_path, trajectory)



