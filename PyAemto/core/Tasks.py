import numpy as np
from PyAemto.core.Task import Task as Task
from pymoo.util.misc import random_permuations
from pymoo.core.population import Population
from pymoo.operators.crossover.binx import BinomialCrossover
# Tpdf = np.array([]) #transfer prob of each task
# Tr = np.array([]) #rewards of inter-task transfer of each task
# Spdf = np.array([]) #selection pdf of each task
# Sr = np.array([]) #selection rewards of each task

class Tasks:
    def  __init__(self, 
        problems, 
        algorithms,
        p_tsf_ub,
        p_tsf_lb,
        MTO,
        alpha,
    ):
        self.tasks = []
        self.ntasks = len(problems)
        self.p_tsf_ub = p_tsf_ub
        self.p_tsf_lb = p_tsf_lb
        self.MTO = MTO
        self.selected_ind = []
        self.alpha = alpha

        #initiallize all tasks
        for i in range(self.ntasks):
            task = Task(problems[i], algorithms[i])
            task.initalize_infill()
            self.tasks.append(task)
        
        self.Spdf = np.full((self.ntasks, self.ntasks), 1.0/(self.ntasks - 1))     #initiallize task selection probability matrix
        self.Spdf = self.Spdf - np.diag(np.full(self.ntasks, 1.0/(self.ntasks - 1)))
        self.Sr = np.full((self.ntasks, self.ntasks), 0.0)
        self.Tpdf = np.full(self.ntasks, (self.p_tsf_ub + self.p_tsf_lb) / 2) #[task1 transfer prob, task2 transfer prob...]
        self.Tr =np.full((self.ntasks,2), 0.0) #[self reward, other reward]
        self.nTsf =  np.full((self.ntasks,2), 0)
    
    def eval_task(self,t,S):
        #select individuals from other tasks
        self.selected_ind = []
        for i,e in enumerate(S[t]):
            if i != t:
                other_task = self.tasks[i]
                ret = other_task.select(e)
                if ret.size > 0:
                    ret.set("skill_factors",i) 
                self.selected_ind.append(ret)
            else:
                self.selected_ind.append(self.tasks[i].get_pop().set("skill_factors",i))

    def reuse(self, t):
        pop =self.selected_ind[t]
        off = Population.empty()
        N = pop.size

        for i,ind in enumerate(self.selected_ind):
            if i != t:
                other_pop = ind
                other_pop_size = other_pop.size
                print(other_pop_size)

                if other_pop_size == 0: 
                    continue
        
                rand_indexs = random_permuations(1,other_pop_size)

                update_num = 0
                mu = [other_pop[rand_indexs[k % other_pop_size]] for k in range(N)]
                parents = list(zip(mu,pop))
                crossover=BinomialCrossover(prob=1.0)
                _off = crossover(self.tasks[t].problem, parents) #2 Parents ==> 2 Children
                off = Population.merge(off,_off)

        if off.size <= 0:
            return 0.0
        rand_indexs = random_permuations(1,off.size)
        off = off[rand_indexs]

        self.tasks[t].eval(off)  #evaluate offsprings

        #Different from the orginal implementation
        pop_F = pop.get("F")
        off_F = off.get("F")
        pop_F = np.reshape(pop_F, (N,1))
        off_F = np.reshape(off_F, (1,off.size))
        off_better = off_F<pop_F

        pop_worse_than=off_better.sum(axis=1)
        off_better_than = off_better.sum(axis=0)

        argsort_pop_worse_than= np.argsort(pop_worse_than)[::-1]
        argsort_better_than = np.argsort(off_better_than)[::-1]
        for i in range(argsort_pop_worse_than.size):
            if pop_worse_than[argsort_pop_worse_than[i]] > 0 or off_better_than[argsort_better_than[i] > 0]:
                pop[argsort_pop_worse_than[i]] = off[argsort_pop_worse_than[i]]
                update_num+=1            
        r_tsf = update_num / N
    
        self.Tr[t,1] =   self.Tr[t,1] * self.alpha + (1 - self.alpha) * r_tsf
    
    # def updateSpdf(self, t):
    #     task_rewards = self.Sr[t]
    #     for i,e in enumerate(self.Spdf[t]):













        
       



        




