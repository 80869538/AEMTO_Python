import numpy as np
import pymoo.gradient.toolbox as anp
from pymoo.core.problem import Problem


class Griewank(Problem):
    def __init__(self, n_var=2):
        super().__init__(n_var=n_var, n_obj=1, xl=0.0, xu=1.0, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = 1 + 1 / 4000 * anp.sum(anp.power(x, 2), axis=1) \
                  - anp.prod(anp.cos(x / anp.sqrt(anp.arange(1, x.shape[1] + 1))), axis=1)

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return np.full(self.n_var, 0)

class Griewank_T(Griewank):
    def __init__(self, M=None, o=None,PF = 0, l=-100, u=100,PS = None ,n_var=50):
        super().__init__(n_var=n_var)
        self.M = M if M is not None else np.identity(self.n_var)
        self.o = o if o is not None else np.full(self.n_var, 0)
        self.PF = PF
        self.PS = PS if PS else np.full(self.n_var, 0)
        self.l =l
        self.u = u
        self.scale_rate_ = (self.u - self.l) / (self.xu - self.xl)
        self.bias_vec_ = np.full((50), self.l)


    
    def _transform(self, x):
        #x: np.array vector
        return np.transpose(np.dot(self.M, np.transpose( self.scale_rate_ * (x - self.xl) + self.bias_vec_ - self.o)))

    
    def _evaluate(self, x, out, *args, **kwargs):
        x = self._transform(x)
        super()._evaluate(x, out, *args, **kwargs)
    
    def _calc_pareto_front(self):
        return self.PF

    def _calc_pareto_set(self):
        return self.PS
