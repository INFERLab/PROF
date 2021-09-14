import numpy as np
import cvxpy as cp
import pdb

class Controller():
    def __init__(self, T, dt, RC_flag = True, **kwargs):
        # dt: planning timestep
        # T: planning horizon
        # RC_flag: Whether using RC model
        # **kwargs: Model Parameters
        self.T = T
        self.RC_flag = RC_flag
        self.err_count = 0
        
        if RC_flag:
        ## RC model: Simulation Study
            self.R = kwargs["R"]
            self.C = kwargs["C"]
            self.Pm = kwargs["Pm"]
            self.eta = kwargs["eta"]
            self.T_sp = kwargs["theta"]
            self.Delta = kwargs["Delta"]
            self.sign = kwargs["sign"] #(+) for heating and (-) for cooling
        else:
        ## ARX model: Hardware-in-the-loop Simulation
            self.ap = kwargs["a"]
            self.bu = kwargs["bu"]
            self.bd = kwargs["bd"]
            self.p = len(self.ap)
            self.m = len(self.bu) # how many u_prev to consider
            self.n_dist = len(self.bd)
            self.Pm = kwargs["Pm"]
            self.T_sp = 75
            self.Delta = 1.8
            
        # Variable
        self.u = cp.Variable(T)
        
        # Save u_i-u_bar from previous time step
        self.u_diff = cp.Parameter(T)
        self.v_bar = cp.Parameter(T)
        self.w_bar = cp.Parameter(T)
        self.objective = cp.sum_squares(self.u-self.u_diff-self.v_bar+self.w_bar)

        ## Info needed for constraints
        if RC_flag:
            self.x0 = cp.Parameter()
            self.d = cp.Parameter(T)
        else:
            ## Expects [x_{t-p}, ..., x_t]
            self.x0 = cp.Parameter(self.p)
            self.d = cp.Parameter((T, self.n_dist))
            
        # Set default value for constraints
        self.u_lower = cp.Parameter(T)
        self.u_lower.value = np.tile(0, T)
        self.u_upper = cp.Parameter(T)
        self.u_upper.value = np.tile(self.Pm, T)
        self.x_lower = cp.Parameter(T)
        self.x_lower.value = np.tile(self.T_sp-self.Delta, T)
        self.x_upper = cp.Parameter(T)
        self.x_upper.value = np.tile(self.T_sp+self.Delta, T)
        

        if RC_flag:
            a = np.exp(-dt/(self.R*self.C))
            b = self.eta * self.R

            lam = np.logspace(1, T, num = T, base = a)
            Lam = np.zeros((T, T))
            for i in range(T):
                for j in range(i+1):
                    Lam[i, j] = a**(i-j)
            B = np.eye(T)*b*(1-a)*self.Pm
            self.d.value = (1-a)*np.tile(32, T)
        else:
            A = np.eye(self.T)
            for i in range(self.T-1):
                A[i+1, max(0, i+1-self.p):i+1] = -np.flip(self.ap)[-(i+1):]
            Lam = np.linalg.inv(A)
    
            lam = np.zeros((self.T, self.p))
            for i in range(self.p):
                lam[i, i:] = np.flip(self.ap)[:self.p-i]
        
            ## note: missing the term on u_{t-1}
            B = np.zeros((self.T, self.T))
            
            for i in range(self.m):
                B += np.diag(np.ones(T-i), -i)*self.bu[i]/self.Pm
            
            self.d.value = np.zeros((T, self.n_dist))
            
        # Constraints
        self.constraints = [-self.u <= -self.u_lower,
                            self.u <= self.u_upper]
        if RC_flag:
            self.constraints += [-Lam@(self.sign*(1-a)*b*self.u+self.d) <= -self.x_lower + lam*self.x0,
                            Lam@(self.sign*(1-a)*b*self.u+self.d) <= self.x_upper - lam*self.x0]
        else:
            self.constraints += [-Lam@(B@self.u + self.d@self.bd + lam@self.x0) <= -self.x_lower,
            Lam@(B@self.u + self.d@self.bd + lam@self.x0) <= self.x_upper]

        self.Problem = cp.Problem(cp.Minimize(self.objective),
                                  self.constraints)
        
    def u_update(self, v_bar, w_bar):
        self.v_bar.value = v_bar
        self.w_bar.value = w_bar
        try:
            self.Problem.solve()
        except:
            print("Solver failed")
            self.u.value = None
            
        ## Check solution valid
        if self.u.value is not None:
            return self.u.value, self.Problem.status
        else:
            u  = (self.x0.value-self.T_sp)/self.Delta
            self.err_count += 1
            return np.ones(self.T)*np.clip(u, 0, 1)*self.Pm, self.Problem.status
    
    def updateState(self, x, u_lower = None, u_upper = None,
                    x_lower = None, x_upper = None,
                    d = None): #
        self.x0.value = x
        
        # Update constraints if necessary
        if u_lower is not None:
            if isinstance(u_lower, int) | isinstance(u_lower, float):
                self.u_lower.value = np.tile(u_lower, self.T)
            else:
                assert len(u_lower) == self.T
                self.u_lower.value = u_lower
        if u_upper is not None:
            if isinstance(u_upper, int) | isinstance(u_upper, float):
                self.u_upper.value = np.tile(u_upper, self.T)
            else:
                assert len(u_upper) == self.T
                self.u_upper.value = u_upper
        if x_lower is not None:
            assert len(x_lower) == self.T
            self.x_lower.value = x_lower
        if x_upper is not None:
            assert len(x_upper) == self.T
            self.x_upper.value = x_upper
            self.T_sp = (x_upper[0]+x_lower[0])/2
            self.Delta = (x_upper[0]-x_lower[0])/2
            
        ## Exog Variables
        if d is not None:
            assert len(d) == self.T
            self.d.value = d


class ControllerGroup():
    def __init__(self, T, dt, parameters, RC_flag = True):
        self.n_agent = len(parameters)
        self.T = T
        self.dt = dt
        self.RC_flag = RC_flag
        self.controller_list = self._init_agents(parameters)
        
    def _init_agents(self, parameters):
        controller_list = []
        for param in parameters:
            controller_list.append(Controller(T = self.T, dt = self.dt, RC_flag = self.RC_flag, **param))
        return controller_list
        
    def updateState(self, x_list, u_list = None, d_list = None, x_lower_list = None, x_upper_list = None):
        for idx, controller in enumerate(self.controller_list):
            controller.updateState(x_list[idx], d = d_list[idx] if d_list is not None else None, x_lower = x_lower_list[idx] if x_lower_list is not None else None, x_upper = x_upper_list[idx] if x_upper_list is not None else None)
            
            ## Initialize the controller with action from prev timestep
            if u_list is not None:
                u_bar = np.mean(u_list, axis = 0)
                controller.u_diff.value = u_list[idx] - u_bar
            else:
                controller.u_diff.value = np.zeros(self.T)
                
    def u_update(self, v_bar, w_bar):
        u_list = []
        #print("v_bar", v_bar)
        #print("w_bar", w_bar.shape)
        for idx, controller in enumerate(self.controller_list):
            #print(idx)
            u_i, status = controller.u_update(v_bar, w_bar)
            if status in ["infeasible", "unbounded"]:
                print(idx, status)
            u_list.append(u_i)
        
        u_bar = np.mean(u_list, axis = 0)
        for idx, controller in enumerate(self.controller_list):
            controller.u_diff.value = u_list[idx] - u_bar
        return u_bar, np.array(u_list)
        
