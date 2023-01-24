import numpy as np
from PyAemto.core.Task import Task as Task
from pymoo.util.misc import random_permuations
from pymoo.core.population import Population
from pymoo.operators.crossover.binx import BinomialCrossover
from copy import copy, deepcopy

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
        pbase
    ):
        self.eps = 1e-10

        self.tasks = []
        self.ntasks = len(problems)
        self.p_tsf_ub = p_tsf_ub
        self.p_tsf_lb = p_tsf_lb
        self.MTO = MTO
        self.selected_ind = []
        self.alpha = alpha
        self.pbase = pbase

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
        self.nTsf =  np.full(self.ntasks, 0)
    
    def eval_other_tasks(self,t,S):
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
                self.selected_ind.append(self.tasks[i].get_pop().set("skill_factors",-1))
    
    def get_tasks_size(self):
        return len(self.tasks)

    def get_result(self,t):
        return self.tasks[t].result()
    
    def get_task_pop(self,i):
        return self.tasks[i].get_pop()

    def eval_task(self,i):
        self.tasks[i].next()
    
    def eval_self_task(self,t):
        pop_tMinus1 = set(self.get_task_pop(t))
        self.eval_task(t)
        pop_t = set(self.get_task_pop(t))
        update_num = self.get_task_pop(t).size - len(pop_tMinus1.intersection(pop_t))
        r_self = update_num/self.get_task_pop(t).size
        print("update_num from eval_self")
        print(update_num)
        self.Tr[t,0] = self.Tr[t,0] * self.alpha + (1 - self.alpha) * r_self

    def reuse(self, t):
        pop =self.selected_ind[t]
        off = Population.empty()
        N = pop.size
        parents_from_other = Population.empty()

        for i,ind in enumerate(self.selected_ind):
            if i != t:
                other_pop = ind
                other_pop_size = other_pop.size

                if other_pop_size == 0: 
                    continue
                rand_indexs = random_permuations(1,other_pop_size)
                mu = [other_pop[rand_indexs[k % other_pop_size]] for k in range(N)]
                parents = list(zip(mu,pop))
                crossover=BinomialCrossover(prob=1.0)
                _off = crossover(self.tasks[t].problem, parents) #2 Parents ==> 2 Children
                _off.set("skill_factors", i)
                off = Population.merge(off,_off)

        # if parents_from_other.size <= 0:
        #     return 0.0
        if off.size <= 0:
            return 0.0

        rand_indexs = random_permuations(1,off.size)
        off = off[rand_indexs]
        self.tasks[t].eval(off)

        # rand_indexs = random_permuations(1,len(parents_from_other))
        # parents_from_other = parents_from_other[rand_indexs]
        # update_num = 0
        # mu = [parents_from_other[rand_indexs[k % len(parents_from_other)]] for k in range(N)]
        # parents = (list(zip(mu,pop)))
        # skill_factors = [i.get("skill_factors") for i in mu]


        # crossover=BinomialCrossover(prob=1.0,n_offsprings=1)
        # _off = crossover(self.tasks[t].problem, parents) #2 Parents ==> 2 Children
        # _off.set("skill_factors",skill_factors)
        # off = Population.merge(off,_off)


        # off.get("F")

        # self.tasks[t].eval(off) 

        #Different from the orginal implementation
        # pop_F = pop.get("F")
        # off_F = off.get("F")


        # for i in range(pop.size):
        #     if pop_F[i] > off_F[i]:
        #         pop[i] = off[i]
        #         pop[i].set("skill_factors",-1)
        #         update_num+=1 

        # r_tsf = update_num / N
        

    
        # self.Tr[t,1] =   self.Tr[t,1] * self.alpha + (1 - self.alpha) * r_tsf
        # pop =self.selected_ind[t]
        # off = Population.empty()
        # N = pop.size

        # for i,ind in enumerate(self.selected_ind):
        #     if i != t:
        #         other_pop = ind
        #         other_pop_size = other_pop.size

        #         if other_pop_size == 0: 
        #             continue
        
        #         rand_indexs = random_permuations(1,other_pop_size)

        #         update_num = 0
        #         mu = [other_pop[rand_indexs[k % other_pop_size]] for k in range(N)]
        #         parents = list(zip(mu,pop))
        #         crossover=BinomialCrossover(prob=1.0,n_offsprings=1)
        #         _off = crossover(self.tasks[t].problem, parents) #2 Parents ==> 2 Children
        #         _off.set("skill_factors",i)
        #         off = Population.merge(off,_off)

        # if off.size <= 0:
        #     return 0.0
        # rand_indexs = random_permuations(1,pop.size)
        # off = off[rand_indexs]

        # self.tasks[t].eval(off)  #evaluate offsprings

        # #Different from the orginal implementation
        pop_F = pop.get("F")
        off_F = off.get("F")
        pop_F = np.reshape(pop_F, (N,1))
        off_F = np.reshape(off_F, (1,off.size))
        off_better = off_F<pop_F

        pop_worse_than=off_better.sum(axis=1)
        off_better_than = off_better.sum(axis=0)
        print("shape")

        print(off_better_than.shape)


        argsort_pop_worse_than= np.argsort(pop_worse_than)[::-1]
        argsort_better_than = np.argsort(off_better_than)[::-1]

        # for i in range(pop.size):
        #     if pop_F[i] > off_F[i]:
        #         pop[i] = off[i]
        #         # pop[i].set("skill_factors",-1)
        #         update_num+=1 
        update_num = 0
        for i in range(argsort_pop_worse_than.size):
            if pop.get("F")[argsort_pop_worse_than[i]] > off.get("F")[argsort_better_than[i]]:
                pop[argsort_pop_worse_than[i]] = off[argsort_pop_worse_than[i]]
                update_num+=1            
        r_tsf = update_num / N
        

    
        self.Tr[t,1] =   self.Tr[t,1] * self.alpha + (1 - self.alpha) * r_tsf
    
    def updateSpdf(self, t):
        task_rewards = self.Sr[t]
        success = self.tasks[t].get_pop().get("skill_factors")
        for i,n in enumerate(self.selected_ind):
            if i != t:
                ns = np.count_nonzero(success == i)
                print("ns:" + str(ns))
                if n.size != 0:
                    task_rewards[i] = (1 - self.alpha) * np.float64(ns/n.size) + task_rewards[i] * self.alpha
        assert(np.diag(self.Sr).sum() == 0) #diag should keep zero, no self transfer
        rewards_sum = np.sum(task_rewards)
        ntasks = self.get_tasks_size()
        pmin = self.pbase / (ntasks - 1)
        self.Spdf[t] = pmin + (1 - (ntasks-1)*pmin) * (task_rewards /(rewards_sum + self.eps))
        self.Spdf[t][t] = 0
        assert(np.diag(self.Spdf).sum() == 0) #diag should keep zero, no self transfer
        self.nTsf[t] += 1
        self.tasks[t].get_pop().set("skill_factors",None)
    

    def update_tsf_pdf(self, t):
        self.Tpdf[t] = self.p_tsf_lb + self.Tr[t,1] * (self.p_tsf_ub - self.p_tsf_lb) / (self.Tr[t,1] + self.Tr[t,0] + self.eps)