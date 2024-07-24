import pandas as pd
import numpy as np
import random
from scipy import stats
import sys

seed = int(sys.argv[1])
np.random.seed(seed)


# need to load eigengenes and reconstructed expression matricees
E01 = np.load('egenes_aa.npy')
X0r, X1r = np.load('Xr_aa.npy', allow_pickle = True)
U = np.load('U_aa.npy')


sorted_egene_indices = np.load('sorted_egene_indices_asthma.npy')
top_egene_indices = np.flip(sorted_egene_indices[-50:])


# load perturbation data
import cmapPy
import cmapPy.pandasGEXpress.parse as parse
Delta_gctx = parse.parse('gene_perturbations_for_ben.gctx')
Delta_gctx.data_df
import pandas as pd
import gzip
# Specify the path to the gzipped text file
file_path = 'mart_export.txt-2.gz'
# Open the gzipped file and read it into a DataFrame
with gzip.open(file_path, 'rt') as file:
    df_genenames = pd.read_csv(file, sep='\t')


gene_dict = np.load('gene_pert_dict.npy', allow_pickle=True)
keys = np.load('gene_keys_asthma_rna_seq.npy', allow_pickle=True)


# match gene indices between disease data and perturbation response data
matching_indices = np.load('matching_indices_asthma_rnaseq.npy')
disease_ind, gene_pert_ind = matching_indices[:,0], matching_indices[:,1]
# use top 10 eigengenes for allergic asthma
estar = U.T[top_egene_indices[:10]][:,disease_ind]
Delta_cc = np.mean(X1r[:,disease_ind] - X0r[:,disease_ind], axis = 0)

# gene perturbation matrix 
A = Delta_gctx.data_df.iloc[:, gene_pert_ind].to_numpy()
A = estar @ A.T


# project onto eigengenes
xF, xI = X0r[:, disease_ind], X1r[:, disease_ind]
xF0, xI0 = xF, xI
xF = np.array([estar@xF0[i] for i in range(len(xF0))])  # Ensure consistent data type
xI = np.array([estar@xI0[i] for i in range(len(xI0))])  # Ensure consistent data type
indI, indF = int(np.random.choice(np.arange(len(xI)))), int(np.random.choice(np.arange(len(xF))))
# print(indI, indF)



import numpy as np
from scipy.optimize import minimize


def objective_function(u, A, D, lam):
    term1 = sum((D - A@u)**2.)**.5 + lam*np.sum(u)
    return term1

# D = estar@Delta_cc

import numpy as np
import copy  # Import the copy module

def optimize(A, D,lam, seed):
    np.random.seed(seed)
    initial_u = np.zeros(len(A[0]))

    D2 = (sum(D**2.))**.5
    constraint = {'type': 'ineq', 'fun': lambda u: np.array([1 - elem for elem in u] + [elem for elem in u])}
    result = minimize(objective_function, initial_u, args=(A, D, lam), constraints=constraint)
    optimal_u = result.x
    R2 = 1. - sum((D - A@optimal_u)**2.)**.5/D2
    magu = np.sum(optimal_u)

    return optimal_u, R2, magu

from multiprocessing import Pool

# Define the function to be parallelized
def process_trial(seed, A, D, lam):
    optimal_u,R2, magu = optimize(A, D,lam, seed=seed)
    return optimal_u, R2, magu

# regularization parameter
lamtab = [.001, .005, .01, .05, .1, .5, 1., 5., 10]

data_for, data_rev = [], []
R2f, R2r = [], []
magu_f, magu_r = [],[]

indI, indF = int(np.random.choice(np.arange(len(xI)))), int(np.random.choice(np.arange(len(xF))))

for i in range(len(lamtab)):
    lam = lamtab[i]
    print('trial ', i, ' of ', len(lamtab))

    D = xI[indI] - xF[indF]
    seed = np.random.choice(np.arange(736353))
    optimal_u,R2, magu = process_trial(seed, A, D, lam)
    data_rev.append(optimal_u)
    np.save('data/uopt_reverse_'+str(indI)+'_'+str(indF), data_rev)

    R2f.append(R2)
    np.save('data/R2f_'+str(indI)+'_'+str(indF), R2f)

    magu_f.append(magu)
    np.save('data/magu_f_'+str(indI)+'_'+str(indF), magu_f)

    D = xF[indF] - xI[indI]
    seed = np.random.choice(np.arange(736353))
    optimal_u, R2, magu = process_trial(seed, A, D, lam)
    data_for.append(optimal_u)
    np.save('data/uopt_forward_'+str(indI)+'_'+str(indF), data_for)

    R2r.append(R2)
    np.save('data/R2r_'+str(indI)+'_'+str(indF), R2r)

    magu_r.append(magu)
    np.save('data/magu_r_'+str(indI)+'_'+str(indF), magu_r)
