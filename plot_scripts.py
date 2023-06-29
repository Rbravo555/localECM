import numpy as np
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import itertools

#plotting variables
colors = ['k', 'm', 'g', 'b', 'c', 'r', 'y']
markers = ['X', '<', '^', '*', 'v', 'P', '>', 'd', '8', 'H']
line_styles = ['-', '--', '-.', ':']

color_cycle = itertools.cycle(colors)
marker_cycle = itertools.cycle(markers)
line_style_cycle = itertools.cycle(line_styles)



def plot_ground_truth(solutions, labels):
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend()
    plt.grid()
    plt.savefig(f'ground_truth.pdf',bbox_inches='tight')
    plt.show()







def plot_original_function_1D(X, A_i, component_to_plot=0):
    number_of_polynomials = len(A_i)

    for i in range(number_of_polynomials):
        color = next(color_cycle)
        marker = next(marker_cycle)
        line_style = next(line_style_cycle)
        try:
            if np.shape(A_i[i])[1]>=1:
                Y = A_i[i][:,component_to_plot]
        except:
            Y = A_i[i]
        Y = A_i[i][:,component_to_plot]
        plt.plot(X, Y , color=color, marker=marker, linestyle=line_style,
                 label=r"$a^{{{}}}(x,{{{}}})$".format(i + 1, i))

    plt.xlabel(r'$x$', size=15, fontweight='bold')
    plt.ylabel(r'$a^{(i+1)}$', size=15, fontweight='bold')

    plt.xticks(rotation='vertical')  # Rotate x-axis tick labels
    plt.rcParams['font.size'] = 8  # Adjust the font size
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.legend()
    plt.grid()
    plt.savefig(f'1d_functions_component_{component_to_plot}.pdf',bbox_inches='tight')
    plt.show()




def plot_basis_functions_1D(X, A_i, component_to_plot=0):
    number_of_polynomials = len(A_i)

    for i in range(number_of_polynomials):
        color = next(color_cycle)
        marker = next(marker_cycle)
        line_style = next(line_style_cycle)
        try:
            if np.shape(A_i[i])[1]>=1:
                Y = A_i[i][:,component_to_plot]
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
    plt.savefig(f'1d_basis_functions_component_{component_to_plot}.pdf',bbox_inches='tight')
    plt.show()




def plot_lp_sparsity(lp_matrices,methods, title ):
    for i in range(len(lp_matrices)):
        # Plot the sparse representation (spy) of the matrix
        plt.subplot(2, 3, i+1)
        plt.spy(lp_matrices[i])
        plt.title(fr'{methods[i]}')
    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.savefig(f'{title}.pdf', bbox_inches='tight')
    plt.show()





def plot_sparsity(short_and_fat_sparse_matrix, title):
    plt.spy(short_and_fat_sparse_matrix)
    plt.xlabel(r'$\bar{\textbf{x}}_j$', size=15, fontweight='bold')
    plt.ylabel(r'$i$-th function',size=15,fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'sparsity_{title}.pdf', bbox_inches='tight')
    plt.show()



