import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, Normal
from copy import deepcopy
import operator
from functools import reduce

import pdb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


### Can move to utils.network if appropriate
class Net(nn.Module):
    def __init__(self, n_bus, n_inverters, shared_hidden_layer_sizes, indiv_hidden_layer_sizes, n_input = 3):
        super(Net, self).__init__()
        #### Multi-headed architecture
        # "Shared" model
        # Set up non-linear network of Linear -> ReLU
        layer_sizes = [n_input * n_bus] + shared_hidden_layer_sizes[:-1]
        layers = reduce(operator.add,
            [[nn.Linear(a,b), nn.ReLU(), ] # nn.BatchNorm1d(b), nn.Dropout(p=0.2)]
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], shared_hidden_layer_sizes[-1])]
        self.base_net = nn.Sequential(*layers)
        
        # Individual inverter model
        layer_sizes = [shared_hidden_layer_sizes[-1]] + indiv_hidden_layer_sizes
        layers = reduce(operator.add,
            [[nn.Linear(a,b),  nn.ReLU(), ] # nn.BatchNorm1d(b), nn.Dropout(p=0.2)]
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], 2)]  # output p and q
        indiv_model = nn.Sequential(*layers)
        self.inverter_nets = nn.ModuleList(
                [deepcopy(indiv_model) for _ in range(n_inverters)]
                )

        # ## Simple fully connected architecture

        # # Set up non-linear network of Linear -> BatchNorm -> ReLU -> Dropout layers
        # self.n_inverters = n_inverters
        # layer_sizes = [4 * n_inverters] + shared_hidden_layer_sizes
        # layers = reduce(operator.add,
        #     [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)]
        #         for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        # layers += [nn.Linear(layer_sizes[-1], 2 * n_inverters)]
        # self.nn = nn.Sequential(*layers)


    def forward(self, state):
        '''
            Input: Vector of voltage magnitudes and angles, real and reactive power demand
            Output: Vector of inverter P setpoints, vector of inverter Q setpoints
        '''

        # Multi-headed architecture
        z = self.base_net(state)
        res = [inverter(z) for inverter in self.inverter_nets]
        Ps = torch.cat([x[:, [0]] for x in res], dim=1)
        Qs = torch.cat([x[:, [1]] for x in res], dim=1)
        return Ps, Qs

        # ## Simple fully connected architecture
        # z = self.nn(state)
        # return z[:, :self.n_inverters], z[:, self.n_inverters:]

class NeuralController(nn.Module):
    def __init__(self, network, memory, lr, lam = 10, scaler = 1000, **env_params):
        super(NeuralController, self).__init__()
        self.nn = network
        self.optimizer = optim.RMSprop(self.nn.parameters(), lr=lr)
        self.lam = lam
        self.memory = memory
        self.mse = nn.MSELoss()
        self.ReLU = nn.ReLU()
        
        n_bus = env_params['n_bus']
        self.gen_idx = env_params['gen_idx']
        self.other_idx = [i for i in range(n_bus) if i not in self.gen_idx]
        
        H = env_params['H']
        R = H[:, :n_bus]
        B = H[:, n_bus:]
        R_new = np.vstack([np.hstack([R[self.gen_idx][:, self.gen_idx],
                                      R[self.gen_idx][:, self.other_idx]]),
                            np.hstack([R[self.other_idx][:, self.gen_idx],
                                       R[self.other_idx][:, self.other_idx]])
                            ])
        B_new = np.vstack([np.hstack([B[self.gen_idx][:, self.gen_idx],
                                      B[self.gen_idx][:, self.other_idx]]),
                            np.hstack([B[self.other_idx][:, self.gen_idx],
                                       B[self.other_idx][:, self.other_idx]])
                            ])
        H_new = np.hstack([R_new, B_new])
        
        self.scaler = scaler
        self.V0 = env_params['V0']
        self.P0 = env_params['P0']
        self.Q0 = env_params['Q0']
        self.V_upper = env_params['V_upper']
        self.V_lower = env_params['V_lower']
        self.S_rating = env_params['S_rating']
        
        # Need to set as nn.Parameter such that to(DEVICE) move these to GPU as well
        self.V0_torch = nn.Parameter(torch.tensor(self.V0).float())
        self.V_upper_torch = nn.Parameter(torch.tensor(self.V_upper).float())
        self.V_lower_torch = nn.Parameter(torch.tensor(self.V_lower).float())
        self.H_torch = nn.Parameter(torch.tensor(H_new).float())
        self.P0_torch = nn.Parameter(torch.tensor(self.P0).float())
        self.Q0_torch = nn.Parameter(torch.tensor(self.Q0).float())
        self.S_rating_torch = nn.Parameter(torch.tensor(self.S_rating).float())

        # Set up projection onto inverter setpoint constraints and linearized voltage constraints
        P = cp.Variable(len(self.gen_idx))
        Q = cp.Variable(len(self.gen_idx))
        
        # P_tilde and Q_tilde are the pre-projection actions
        P_tilde = cp.Parameter(len(self.gen_idx))
        Q_tilde = cp.Parameter(len(self.gen_idx))
        
        # No inverter buses
        P_nc = cp.Parameter(len(self.other_idx))
        Q_nc = cp.Parameter(len(self.other_idx))
        
        P_av = cp.Parameter(len(self.gen_idx))
        
        # Voltage: Apply to All Buses
        z = cp.hstack([P, P_nc, Q, Q_nc]) # z: (70, )
        constraints = [self.V_lower - self.V0 <= H_new@z,
                       H_new@z <= self.V_upper - self.V0]
        
        ## Power: Only applies to Inverters
        PQ = cp.vstack([self.P0[self.gen_idx] + P,
                        self.Q0[self.gen_idx] + Q]) # (2, n)
        constraints += [0 <= self.P0[self.gen_idx] + P,
                       self.P0[self.gen_idx] + P <= P_av,
                       cp.norm(PQ, axis = 0) <= self.S_rating]
        
        objective = cp.Minimize(cp.sum_squares(P - P_tilde) + cp.sum_squares(Q - Q_tilde))
        problem = cp.Problem(objective, constraints)

        self.proj_layer = CvxpyLayer(problem, variables=[P, Q],
                parameters=[P_tilde, Q_tilde,
                           P_nc, Q_nc, P_av])
        
        self.proj_count = 0
        
    def forward(self, state, Sbus, P_av, inference_flag = True):
        '''
        Input:
            state: [dV(k-1), P_nc, Q_nc]
          where,
                Z_nc = Z - Z0
            May get (n, dim) or (dim);
        Output:
            P, Q (with repsect to the reference point)
        '''
        ## Get information for non-controllable loads
        P_all = Sbus.real /self.scaler
        Q_all = Sbus.imag /self.scaler
        if len(Sbus.shape)==1:
            P_nc = Sbus.real[self.other_idx] / self.scaler
            Q_nc = Sbus.imag[self.other_idx] / self.scaler
        elif len(Sbus.shape)==2:
            P_nc = Sbus.real[:, self.other_idx] / self.scaler
            Q_nc = Sbus.imag[:, self.other_idx] / self.scaler
        else:
            print("Well, not expected to happen")

        P_tilde, Q_tilde = self.nn(state.to(DEVICE)) # n x n_inverter
        
        ## During inference if the action is already feasible, not need to project
        if inference_flag:
            P_tilde = P_tilde.squeeze()
            Q_tilde = Q_tilde.squeeze()
            if self.is_feasible(P_tilde.detach().clone()/self.scaler,
                               Q_tilde.detach().clone()/self.scaler,
                               P_nc, Q_nc, P_av):
                P_all[self.gen_idx] = P_tilde.detach().cpu().numpy() / self.scaler
                Q_all[self.gen_idx] = Q_tilde.detach().cpu().numpy() / self.scaler
                return P_all, Q_all
            else:
                try:
                    P, Q = self.proj_layer(P_tilde/self.scaler, Q_tilde/self.scaler,
                        torch.tensor(P_nc).float().to(DEVICE),
                        torch.tensor(Q_nc).float().to(DEVICE),
                        torch.tensor(P_av).float().to(DEVICE))
                    self.proj_count += 1
                    P_all[self.gen_idx] = P.detach().cpu().numpy()
                    Q_all[self.gen_idx] = Q.detach().cpu().numpy()
                except: # The solver dies for some reason
                    P_all[self.gen_idx] = 0
                    Q_all[self.gen_idx] = 0
                return P_all, Q_all
        else:
            #pdb.set_trace()
            P, Q = self.proj_layer(P_tilde/self.scaler, Q_tilde/self.scaler,
                        torch.tensor(P_nc).float().to(DEVICE),
                        torch.tensor(Q_nc).float().to(DEVICE),
                        torch.tensor(P_av).float().to(DEVICE))
            proj_loss = self.mse(P.detach(), P_tilde/self.scaler)  \
                        + self.mse(Q.detach(), Q_tilde/self.scaler)
            return P, Q, proj_loss
    
    def update(self, batch_size = 64, n_batch = 16):
        for _ in range(n_batch):
            state, Sbus, P_av = self.memory.sample_batch(batch_size = batch_size)
            P, Q, proj_loss = self.forward(state, Sbus, P_av, inference_flag = False)
            #pdb.set_trace()
            curtail = self.ReLU(torch.tensor(P_av).to(DEVICE) - P)
            loss = curtail.mean() + self.lam * proj_loss
            print(f'curtail = {curtail.mean().item()}, proj_loss = {proj_loss.item()}')
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
               
    def is_feasible(self, P, Q, P_nc, Q_nc, P_av):
        '''
         Input: P, Q (n_bus)
        '''
        eps = 1e-6
        assert P.ndimension() == 1

        z = torch.cat([P, torch.tensor(P_nc).float().to(DEVICE),
                       Q, torch.tensor(Q_nc).float().to(DEVICE)], dim = -1) # (70)
        v = self.H_torch.matmul(z) # (35)
        
        if torch.any(v < self.V_lower_torch -self.V0_torch - eps) | torch.any(v > self.V_upper_torch-self.V0_torch+eps):
            return False

        P = P + self.P0_torch[self.gen_idx]
        Q = Q + self.Q0_torch[self.gen_idx]
        PQ = torch.stack([P, Q]) # (2, 21)
        if torch.any(torch.norm(PQ, dim = 0) > self.S_rating_torch + eps):
            return False

        if torch.any(P < 0-eps) | torch.any(P > torch.tensor(P_av).to(DEVICE)+eps):
            return False
        else:
            return True

   
    


# if __name__ == "__main__":
#     # Toy test code
#     n = 3
#     nbatch = 2
#     model = Net(n, [200,200,40], [30,5])
#     controller = NeuralController(n, model, np.random.randn(n, 2*n), 0.95, 1.05)
#     action = controller(torch.randn(nbatch, 4*n), torch.randn(nbatch, n), torch.randn(nbatch, n))
        

