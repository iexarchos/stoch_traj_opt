"""
Created on Tue Feb 18 14:50:12 2020

@author: yannis
"""

from multiprocessing import Process, Pipe
import os
import numpy as np
import time
#from pdb import set_trace as bp



class StochTrajOptimizer:
    def __init__(self,
                 env,
                 sigma = 0.2,
                 kappa = 1,
                 alpha = 1,
                 Num_processes = 12,
                 Traj_per_process = 30,
                 TimeSteps = 200,
                 Iterations = 20,
                 seed = None,
                 render = True,
                 initial_guess = None,
                 **kwargs):        

        self.sigma = sigma  # noise intensity
        self.kappa = kappa # transformation "temperature"
        self.alpha = alpha # learning rate
        self.num_process = Num_processes # number of processes for parallel computation
        self.Traj_per_process = Traj_per_process # number of trajectories simulated by each process
        self.N = TimeSteps # trajectory time horizon
        self.ITER = Iterations # number of optimization iterations
        self.M = Num_processes*Traj_per_process # total number of trajectories simulated
        self.seed = seed # seed, in case the environment reset is stochastic
        self.render = render # whether or not to render the execution of each iteration output
        self.initial_guess = initial_guess # location of .npy file containing initial control sequence guess (can be a sequence obtained by previous run of algorithm)
        self.env = env # environment
        self.kwargs = kwargs # environment arguments


    ###############################
    ####### Child process  ########
    ###############################
    def child_process(conn):
            print('simulation process id:', os.getpid())          
            # Receive the initialization data
            initialization_data = conn.recv()
            print('Initialization data: ', initialization_data)
            N,sigma,n_traj,seed,env_fn,kwargs = initialization_data # number of trajectories simulated by each process
            sim = env_fn(renders=False, **kwargs)
            if seed is not None:
                sim.seed(seed) # it is assumed that the seed is controlled by a method called "seed" in the environment
            sim.reset()
            while True:
                command_and_args = conn.recv()   # Get command from main process
                if command_and_args[0] == "control":    # Run trajectories
                    u = command_and_args[1]
                    np.random.seed([os.getpid()])
                    E = sigma*np.random.randn(N,u.shape[1],n_traj)
                    J = np.zeros((n_traj,))
                    for i in range(n_traj):
                        if seed is not None:
                            sim.seed(seed)
                        sim.reset()
                        cost = 0
                        for j in range(N):
                            v = u[j,:] + E[j,:,i]
                            state, c, done, _ = sim.step(v)
                            c=-c #reverse sign to make it a cost instead of a reward
                            cost+=c
                        J[i] = cost   
                #eprint('Done: sim process #',os.getpid())          
                if command_and_args[0] == "get_J":     # Get all the Js in the buffer
                    conn.send([J,E])
                # print('sent cost')
                if command_and_args[0] == "stop":       # Stop the program
                    conn.close()
                    break





    ###############################
    ######## Main process  ########
    ###############################
    def optimize(self):
        print('main program process id:', os.getpid())  
        # Setup and store all processes
        processes = []
        for i in range(self.num_process):
            # Create a pair of pipes that can communicate with each other through .send() and .recv()
            parent_conn, child_conn = Pipe()
            # Create the sub-process and assign the pipe to it
            p = Process(target=StochTrajOptimizer.child_process, args=(child_conn, ))
            processes.append([p, parent_conn])
            # Start the process
            p.start()
            # Send the initial arguments for initialization
            parent_conn.send([self.N,self.sigma,self.Traj_per_process,self.seed,self.env,  self.kwargs])

        self.world = self.env(renders=self.render, **self.kwargs)
        self.ctrl_dim = self.world.action_space.shape[0]
        if self.initial_guess is not None:
            u = np.load(self.initial_guess)
            print('Initialized control with existing control sequence')
        else:
            u = np.zeros((self.N,self.ctrl_dim)) # initialize control sequence with zeros
            print('Initialized control with 0')
        
        # Optimization loop
        self.uopt = u.copy() # uopt keeps track of the optimal control sequence
        self.Jopt = np.inf # initialization of optimal cost
        print('Starting optimization...')
        start_time = time.time()
        for iter_id in range(self.ITER):
            J = [];
            E = [];
            for i in range(self.num_process):
                # Get the control signal for this trajectory
                processes[i][1].send(["control", u])    # Ask process i to simulate the control signals u
 
            for i in range(self.num_process): 
                processes[i][1].send(["get_J"])     # Get the trajectory costs for the trajectories that process i was asked to simulate
                Js_i, E_i = processes[i][1].recv()
                J.append(Js_i)
                E.append(E_i)
                #print('got J,E,')

            J = np.concatenate(J)
            E = np.concatenate(E,axis=2)
            J = J - min(J)
            S = np.exp(-self.kappa*J)
            norm = np.sum(S)
            S = S/norm
            #print(max(S))
            # calculate new control
            for j in range(self.N):   # update controls
                for k in range(self.ctrl_dim):
                    u[j,k] = u[j,k] + self.alpha*np.sum(S*E[j,k,:])
            end_time = time.time()
            
            #print('executing control...')
            Jcur = self.replay_traj(u) #execute trajectory to evaluate control and get current cost
        
            if Jcur < self.Jopt:   # compare current cost with optimal
                self.uopt = u.copy()
                #Xopt[:,:] = Xnew[:,:]
                self.Jopt = Jcur.copy()
            print("Iteration %.0f took %.2f seconds, Current reward: %.2f, Optimal reward %.2f" % (iter_id+1, (end_time - start_time), -Jcur, -self.Jopt))
            start_time = time.time()
            
        # Wrap up
        for i in range(self.num_process):
            Js_i = processes[i][1].send(["stop"])
            processes[i][0].join()
            
        print('Optimization completed.')
        return self.uopt, -self.Jopt
    
    
    
    
    
    
    
    def replay_traj(self,u):
        if self.seed is not None:
            self.world.seed(self.seed) # it is assumed that the seed is controlled by a method called "seed" in the environment
        self.world.reset()
        J = 0
        for j in range(self.N):
                state, c, done, _ = self.world.step(u[j,:])
                c = -c #reverse sign to make it a cost instead of a reward
                J += c
                if self.render:
                    time.sleep(3*1.0/240.0)
        return J        
        
    
    
    
    
    
    
    
    
    
    
    
    
    


if __name__ == '__main__':
    
    from franka_pybullet_envs.Franka_scoop_env_v0 import FrankaArmScoopEnv
    
    sess = StochTrajOptimizer(env=FrankaArmScoopEnv, sigma = 0.1,initial_guess ='optimal_action_sequences/u_opt.npy')
    uopt,Jopt = sess.optimize()
   
    # save uopt?
    # np.save('optimal_action_sequences/u_opt.npy',uopt) 
    
    #replay optimal trajectory?
    sess.replay_traj(uopt)
            
