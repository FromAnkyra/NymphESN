import numpy as np
import scipy.sparse as sparse
from . errorfuncs import ErrorFuncs
import sys
np.set_printoptions(threshold=sys.maxsize)

class NymphESN():
    """
    ESN implementation.

    class that implements ESNs with various parametres, topologies, and update styles.
    Initialising, training, and testing the ESN are all supported.
    currently training is only done using the pseudoinverse method.
    """
    def __init__(self, K: int, N: int, L: int, seed: int, f=np.tanh, rho=2, density=0.1):
        """
        Initialise the ESN with random input and inner weights.

        K: number of input nodes.
        N: number of reservoir nodes.
        L: number output nodes.
        f (default numpy.tanh): update function
        rho (default 2): spectral radius
        """ 
        np.random.seed(seed)

        self.K = K
        self.N = N
        self.L = L

        self.f = f
        self.rho = rho
        self.density = density 

        # self.u = np.random.rand(K, 1) # input vector
        # self.x = np.zeros(N, 1) # state vector
        # self.v = np.zeros(L, 1) # output vector

        self.Wu : np.array #input weights - shape (N, K)
        self.W : np.array # reservoir weights - shape (N, N)
        self.Wv : np.array # output weights, to be trained - shape (L, N)

        self.TWashout = 0 # number of Washout runs
        self.TTrain = 0 # number of Training runs
        self.TTest = 0 # number of Test runs
        self.TAll = self.TWashout + self.TTrain + self.TTest

        self.set_weights() # set random inner weights
        self.set_input_weights() # set random input weights

    def set_input_weights(self, Wu=None):
        '''
        Set the input weights to Wu if given, and to random values otherwise.
        
        Wu: np.array - shape = (N, K)
        '''
        if Wu is not None:
            Wu.shape = (self.N, self.K)
            self.Wu = Wu
        else: 
            self.Wu = np.random.uniform(-1, 1, size=(self.N, self.K))
        return

    def set_weights(self, W=None, density=0.1):
        '''
        Set inner weights to W if given, and to random weights otherwise

        W: np.array - shape = (N, N)
        '''
        if W is not None:
            W.shape = (self.N, self.N)
            self.W = W
        elif W is None:
            W = sparse.random(self.N, self.N, density=self.density)
            W.data = (W.data - 0.5) * 2
            W = W.toarray()
            s = np.linalg.svd(W, compute_uv=False)
            self.W = W / s[0]

            # svd = np.linalg.svd(self.W, compute_uv=False)
            # self.W = self.W/svd[0]
        return

    def set_data_lengths(self, TWashout: int, TTrain: int, TTest: int):
        self.TWashout = TWashout
        self.TTrain = TTrain
        self.TTest = TTest
        self.TAll = TWashout + TTrain + TTest
    
    def set_input_stream(self, uall):
        uall.shape = (self.K, -1) # infers TAll from the length of uall
        self.uall = uall
        return
    
    def run_timestep(self, t: int): # this is currently in the "flat" configuration, but would be easy to modify
        # print(t)
        # print(self.xall)
        # print(self.W)
        u_t = self.uall[:, t]
        x_t = self.xall[:,t]
        # print(x_t)
        x_t.shape = (self.N, 1)
        Wu_x_u = self.Wu.dot(u_t)
        Wu_x_u.shape = (self.N, 1)
        # print(f"x(t): {x_t.shape}")
        # print(f"Wu.u(t+1): {Wu_x_u.shape}")
        # print(f"rho: {self.rho}")
        x_t1 = self.f(self.rho * x_t.T.dot(self.W) + Wu_x_u.T)
        # print(x_t1)
        # print(f"xall: {self.xall.shape}, x_t1: {x_t1.shape}")
        self.xall = np.hstack((self.xall, x_t1.T))
        # print(f"xall: {self.xall.shape}")
        return

    def run_full(self, W=[], Wu=[]):
        if self.TAll > self.uall.shape[1]:
            raise ValueError("not enough inputs. Try set_input_stream with a longer input stream?")
        if W == []:
            W = [self.W]
        if Wu == []:
            Wu = [self.Wu]
        self.xall = np.zeros((self.N, 1)) # set initial x0 state to 0
        for t in range(self.TAll-1):
            # print((t)%len(W))
            self.W = W[(t)%len(W)]
            # print(self.W)
            self.Wu = Wu[(t)%len(Wu)]
            self.run_timestep(t)
            # print(f"t: {t}")
        return
    
    def train_reservoir(self, vtarget: np.array): #TODO: ????
        vtarget.shape = (self.TTrain, self.L)
        M = self.xall[:, self.TWashout:self.TWashout+self.TTrain] #shape(N, T-1)
        M.shape = (self.N, self.TTrain)
        M = np.transpose(M)
        M_plus = np.linalg.pinv(M) 
        self.Wv = M_plus.dot(vtarget).T
        # print(f"Wv shape: {self.Wv.shape}")
        return

    def get_output(self):
        # print(f"Wv: {self.Wv.shape}\nxall: {self.xall.shape}")
        # print(f"Wv shape: {self.Wv.shape}")
        # print(f"xall shape: {self.xall.shape}")
        self.vall = self.Wv.dot(self.xall)
        # for v in range(self.TAll):
        #     if self.vall[:, v] > 2 or self.vall[:, v] < -2:
        #         print(f"t={v}, v={self.vall[:, v]}")
        # print(self.vall.size)
        return
        # print(f"vall: {self.vall.shape}")

    def get_error(self, vtarget: np.array, error_func = ErrorFuncs.nrmse):
        
        # for i in range(self.TTrain + self.TTest):
        #     print(vtarget[i])
        #     print(self.vall[:,i])
        vtarget.shape == (self.TAll, self.L)
        vtest = self.vall[:,self.TWashout+self.TTrain:]
        testtarget = vtarget[self.TWashout+self.TTrain:-1]
        test_error = error_func(vtest, testtarget)
        
        vtrain = self.vall[:,self.TWashout:self.TWashout+self.TTrain]
        traintarget = vtarget[self.TWashout:self.TWashout+self.TTrain]
        train_error = error_func(vtrain, traintarget)

        return train_error, test_error
    
    # def reset():
    #     #zero the input stream and the state

    






    
