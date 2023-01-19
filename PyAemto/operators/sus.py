from pymoo.core.selection import Selection

class SUS(Selection): 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _do(self, _, pop, n_select, n_parents=1, **kwargs):
        
