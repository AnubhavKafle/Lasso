def LASSO(geno, expr, alpha):
    path = "/home/anubhavk/Desktop/Masters_work/Trans-eQTL_Discovery_Methods/TEJAAS_LASSO/main"
    clibs = np.ctypeslib.load_library('../lib/mlpack_lasso_o_test.so', path)
    clarss = clibs.cvfold2
    clarss.restype = ctypes.c_bool
    clarss.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
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


    snp_tejas = geno.reshape(-1,)
    genes_tejas = expr.reshape(-1,)
    nsnps_t = geno.shape[0]
    nsample = geno.shape[1]
    ngene_t = expr.shape[0]
    LLR_snp = np.zeros(nsnps_t)
    model_betas = np.zeros(ngene_t)
    mse_values = np.zeros(alphas.shape[0]*nsnps_t)
    best_alpha = np.zeros(nsnps_t)
    success = clarss(snp_tejas,genes_tejas, ngene_t, nsample, alphas, alphas.shape[0], mse_values, best_alpha, model_betas, LLR_snp, nsnps_t)
    del LLR_snp
    #model_betas = model_betas.reshape(nsnps_t,ngene_t)
    return model_betas

def cpvalcomp(geno, expr, qcal):
    _path = "/home/anubhavk/Desktop/Masters_work/Trans-eQTL_Discovery_Methods/TEJAAS_LASSO/main"
    clib = np.ctypeslib.load_library('../lib/linear_regression.so', _path)
    cfstat = clib.fit
    cfstat.restype = ctypes.c_int
    cfstat.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                       np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED'),
                       ctypes.c_int,
                       ctypes.c_int,
                       ctypes.c_int,
                       np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS, ALIGNED')
                      ]

    x = geno.reshape(-1,)
    y = expr.reshape(-1,)
    xsize = x.shape[0]
    nsnps = geno.shape[0]
    nsample = geno.shape[1]
    ngene = expr.shape[0]
    fstat = np.zeros(nsnps * ngene)
    success = cfstat(x, y, nsnps, ngene, nsample, fstat)
    pvals = 1 - stats.f.cdf(fstat, 1, nsample-2)
    pvals = pvals.reshape(nsnps, ngene)
    qscores = np.array([qstat.qscore(pvals[i,:]) for i in range(nsnps)])
    #pqvals  = np.array([qstat.p_qscore(qscores[i], qcal) for i in range(nsnps)])
    gene_indices = list()
    snp_indices = list()
    genes_tejas = list()
    for snp in range(nsnps):
        if qscores[snp] > 46.0:  #cutting off at this point of 0.01 significance on ECDF model
            snp_indices.append(snp)
            gene_indices.append(np.where(pvals[snp, :] < 0.01)[0])
    snp_tejas = geno[np.array(snp_indices),:]
    for i,j in enumerate(gene_indices):
        genes_tejas.append(np.array(expr[j,:]))
    print(qscores[snp_indices[0]])   
    alphas = np.array([30,31],dtype = "float64")
    qscores_refined = list()
    pqscores_refined = list()
    for i,snp in enumerate(snp_indices):
        snp_tejas = geno[snp,:]
        model_betas = LASSO(np.array([snp_tejas]), genes_tejas[i], alphas)
        del_genes = np.where(model_betas== 0)[0]
        del_genes_indices = np.array([gene_indices[i][x] for x in del_genes])
        pvals_refined = pvals[snp,[x for x in range(pvals.shape[1]) if x not in del_genes_indices]]
        qscores_refined.append(qstat.qscore(pvals_refined))
        #print(qscores_refined)
        pqscores_refined.append(qstat.p_qscore(qscores_refined[i],qcal))
    gene_refined_indices = list()   
    for i,snp in enumerate(snp_indices):
        if pqscores_refined[i] <= 0.01:
            gene_refined_indices.append(np.where(pvals[snp,:] <= 0.01)[0])
        else:
            gene_refined_indices.append(np.array([],dtype=int))

            
    return pvals, qscores_refined, pqscores_refined, gene_refined_indices, snp_indices
