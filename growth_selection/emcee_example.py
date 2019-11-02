import matplotlib.pyplot as plt
import numpy as np

import lmfit

x = np.linspace(1, 10, 250)
np.random.seed(0)
y = 0.3 + 0.2 * x + np.random.randn(x.size)
plt.plot(x, y, 'b')

p = lmfit.Parameters()
p.add_many(('alpha', 0.5), ('beta', 0.3))

def residual(p):
    v = p.valuesdict()
    return (v['alpha'] + v['beta'] * x) - y

mi = lmfit.minimize(residual, p, method='nelder', nan_policy='omit')
lmfit.printfuncs.report_fit(mi.params, min_correl=0.5)

plt.plot(x, y, 'b')
plt.plot(x, residual(res.params) + y, 'r')
plt.legend(loc='best')
plt.show()


mi.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))

res = lmfit.minimize(residual, method='emcee', nan_policy='omit', burn=300, steps=1000, thin=20,
                     params=mi.params, is_weighted=False)

res.bic
res
