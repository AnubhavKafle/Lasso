import numpy as np
#from mpi4py import MPI
import ctypes
from scipy import stats
import os
import time
import argparse
import math
from dosageparser import DosageParser
import output

def parse_args():

    parser = argparse.ArgumentParser(description='Trans-Eqtls from Joint Association AnalysiS (TEJAAS)')

    parser.add_argument('--genotype',
                        type=str,
                        dest='genotype_filename',
                        metavar='FILE',
                        help='input genotype file')

    parser.add_argument('--sample',
                        type=str,
                        dest='sample_filename',
                        metavar='FILE',
                        help='input fam file')

    parser.add_argument('--expression',
                        type=str,
                        dest='expression_filename',
                        metavar='FILE',
                        help='input expression file')

    parser.add_argument('--output',
                        type=str,
                        dest='output_fileprefix',
                        metavar='FILE',
                        help='output file prefix')

    parser.add_argument('--start',
                        type=int,
                        dest='startsnp',
                        help='starting SNP index')

    parser.add_argument('--end',
                        type=int,
                        dest='endsnp',
                        help='ending SNP index')

    opts = parser.parse_args()
    return opts


def read_expression(filename):
    gene_names = list()
    expression = list()
    with open(filename, 'r') as mfile:
        header = mfile.readline()
        donorids = header.strip().split()[1:]
        for line in mfile:
            linesplit = line.strip().split()
            expression.append(np.array(linesplit[1:], dtype=float))
            gene_names.append(linesplit[0])
    expression = np.array(expression)
    return donorids, expression, gene_names

def norm_binom(gt, freq):
    f = freq.reshape(-1, 1)
    gt = (gt - (2 * f)) / np.sqrt(2 * f * (1 - f))
    return gt


def mlpack_lasso_cv(geno, expr, alphas):
    _path = "/home/anubhavk/Desktop/thesis_work/tejaas_required_data/LASSO_RG/saikait_lib/"
    clib = np.ctypeslib.load_library('mlpack_lasso_o_test.so', _path)
    clars = clib.cvfold2
    clars.restype = ctypes.c_bool
    clars.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                       np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                       ctypes.c_int,
                       ctypes.c_int,
                       np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                       ctypes.c_int,
                       np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                       np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                       np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                       np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                       ctypes.c_int,
                       ]

    nsnps   = geno.shape[0]
    nsample = geno.shape[1]
    ngene   = expr.shape[0]
    LLR_snp = np.zeros(nsnps)
    y = geno.reshape(-1,)
    X = expr.transpose().reshape(-1,)
    model_betas = np.zeros(nsnps*ngene)
    print (nsnps * ngene)
    mse_values = np.zeros(alphas.shape[0]*nsnps)
    best_alpha = np.zeros(nsnps)
    success = clars(y,X, ngene, nsample, alphas, alphas.shape[0], mse_values, best_alpha, model_betas, LLR_snp, nsnps)
    model_betas = model_betas.reshape(nsnps,ngene)
    paranum = [np.where(model_betas[i,:] != 0)[0].shape[0] for i in range(nsnps)]
    pvals = np.array([(1 - stats.chi2.cdf(i, paranum[j])) for j,i in enumerate(LLR_snp)])
    pvals = np.array([1 if math.isnan(i) else i for i in pvals])
    snp_qualified = np.array(np.where(pvals <= 0.001)[0])
    print(snp_qualified.shape[0],pvals)
    snp_index = list()
    for i in range(ngene):
        snp_index.append(snp_qualified[np.where(model_betas[snp_qualified,i] != 0)[0]])
    return snp_index,model_betas,best_alpha

opts = parse_args()
out_fileprefix = opts.output_fileprefix
genotype_filename = opts.genotype_filename
sample_filename = opts.sample_filename
expression_filename = opts.expression_filename
startsnp = opts.startsnp
endsnp = opts.endsnp

start_time = time.time()

ds = DosageParser(genotype_filename, sample_filename, startsnp, endsnp)
dosage = ds.dosage
snpinfo = ds.snpinfo
donorids = ds.sample_id
nsnps = ds.nsnps
nsample = ds.nsample

sampleids, expression, gene_names = read_expression(expression_filename)

choose_ids = [x for x in sampleids if x in donorids]
dosage_indices = [i for i, x in enumerate(donorids)  if x in choose_ids]
exprsn_indices = [i for i, x in enumerate(sampleids) if x in choose_ids]

geno = dosage[:, dosage_indices]
freq = np.array([x.freq for x in snpinfo])

print ("Completed data reading")

# These are the inputs to mlpack_lasso_cv function 
geno = norm_binom(geno, freq)
expr = expression[:, exprsn_indices]
#p = np.random.choice(20000,1000)
#expr  = expr[p,]
print(expr.shape,geno.shape)
print ("Normalization complete")

alphas = np.array([30],dtype = "float64")
#alphas = np.array([0.005,0.1,0.5,1.5,1.0])
snp_index,best_betas, best_alpha = mlpack_lasso_cv(geno, expr, alphas)

#print(snp_index)


#snp_index = list()
with open("pp.txt","w") as output:
    for i, j in enumerate(snp_index):
        if j.shape[0] == 0:
            outputline = gene_names[i]+"\t"+"NA"
        else:
            outputline = gene_names[i]+"\t"+ ",".join([snpinfo[x].rsid for x in j])

        output.write(outputline+"\n")

with open("beta.txt","w") as output1:
    for i in range(best_betas.shape[0]):
        writeline = " ".join([str(x) for x in best_betas[i,:]])
        output1.write(writeline+"\n")

print ("time taken was :", time.time() - start_time)

"""
print("best lambdas were:",best_alpha)

import matplotlib.pyplot as plt
plt.figure()
m_log_alphas = -np.log10(alphas)
#ymin, ymax = 2300, 3800
plt.plot(m_log_alphas, mse_values, 'k', linewidth=2)
plt.axvline(-np.log10(best_alpha), linestyle='--', color='k',
            label='alpha: CV estimate')

plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error at each alpha: C++')
plt.axis('tight')
#plt.ylim(ymin, ymax)

plt.savefig("snp_msv_bestAlpha_50.pdf")
"""
