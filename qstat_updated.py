def qscore(pvals):
    p = np.sort(pvals)
    n = p.shape[0]
    kmax = min(100, n)
    krange = [i + 1 for i in range(kmax)]
    digamma_n1 = special.digamma(n + 1)
    z = - ( np.log(p[:kmax]) - (special.digamma(krange) - digamma_n1) )
    zsum = np.cumsum(z)
    res = np.max(zsum)
    return res

def q_calibrate(location,p0=0.001):
    qneg = np.load(location)
    qneg_sorted = qneg[np.argsort(qneg)]
    S = qneg_sorted.shape[0]
    ntop = int(p0 * S)
    #cdf = np.arange(S)/float(S - 1)
    matchindex = S - ntop - 1
    qjoin = qneg_sorted[ matchindex  ]
    qcalc = qneg_sorted[ matchindex: ]
    lam = ntop / np.sum(qcalc - qjoin) # maximum likelihood estimate of exponential distribution of Q
    qcal = Q_Cal(join = qjoin,
                 lam  = lam,
                 coef = ntop / (S - 1),
                 ecdf = ECDF(qneg_sorted[::-1]))
    #print(ntop)
    return qcal

def p_qscore(qscore, qcal):
    if qscore > qcal.join:
        res = qcal.coef * np.exp( - (qscore - qcal.join) * qcal.lam)
    else:
        res = 1 - qcal.ecdf(qscore)
    return res
