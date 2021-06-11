from tensorflow.keras.callbacks import Callback
import numpy as np
from scipy import sparse

class Norm_Constraint(Callback):
    
    def __init__(self, model, Ad, K, N, layers=[2,3], withConstraint=False, applyFista=True):
        # Whether has Lipschitz Constraint
        self.withConstraint = withConstraint
        # Whether using the accelerated version Fista
        self.applyFista = applyFista
        self.model = model
        self.Ad = Ad
        # Number of samples
        self.K = K
        # [Number of features per sample, Nums of neurons for every layer]
        self.N = N 
        self.layers = layers
        # number of iteration to update weight matrix in one step
        self.nit = 10
        self.rho = 5.0 # theta_bar
        self.criterion = 100
        self.cnst = 0.01

    def get_mask(self, Ad, index_weight):
        adjency = Ad + np.eye(self.K, self.K) # This line is modified by CHEN
        unity = np.ones((self.N[index_weight - 1], self.N[index_weight - 2]))
        M = np.kron(adjency, unity)
        return M


    def get_projection(self, w, Ad, index_weight):
        if type(Ad) is sparse.csr_matrix:
            Ad = Ad.toarray()
        # Element-wise multiplication
        return np.multiply(w, self.get_mask(Ad, index_weight))


    def Constraint(self, w, A, B, index):
        # np.spacing(1) == 2.220446049250313e-16, gam is a scalar
        gam = 1.99 / (np.square(np.dot(np.linalg.norm(A, ord=2), np.linalg.norm(B, ord=2)) + np.spacing(1)))
        Y = np.zeros([self.model.layers[self.layers[-1]].output_shape[1], self.model.layers[self.layers[0]].input_shape[1]]) # This line is modified by CHEN
        # print("Y",Y.shape)
        for _ in range(self.nit):
            # KEY POINT REDUCE SPECTRAL NORM
            w_new = w - (np.transpose(A) @ Y @ np.transpose(B))
            w_new[w_new < 0] = 0.0 # ensure non-negative weights # This line is modified by CHEN
            w_new = self.get_projection(w_new, self.Ad, index) # This line is modified by CHEN

            T = A @ w_new @ B # T.shape == Y.shape
            [u,s,v] = np.linalg.svd(T)
            # print(u.shape, s.shape, v.shape)
            criterion = np.linalg.norm(w_new - w, ord='fro')
            constraint = np.linalg.norm(s[s > self.rho] - self.rho, ord=2) # L2 norm sqrt(a.T*a) # rho = 1
            Yt = Y + gam * T
            [u1, s1, v1] = np.linalg.svd(Yt / gam, full_matrices=False)
            s1 = np.clip(s1, 0, self.rho)
            # print("s1", s1.shape)
            Svd_recons = u1 @ np.diag(s1) @ v1 # This line is modified by CHEN 
            Y = Yt - gam * Svd_recons
            if (criterion < criterion and constraint < self.cnst): 
                return w_new
        return w_new


    def Constraint_Fista(self, w, A, B, index):
        Y0 = np.zeros([self.model.layers[self.layers[-1]].output_shape[1], self.model.layers[self.layers[0]].input_shape[1]]) # This line is modified by CHEN
        Y = Y0
        Yold = Y0
        gam = 1 / ((np.linalg.norm(A, ord=2) * np.linalg.norm(B, ord=2) + np.spacing(1)) ** 2)
        alpha = 2.1

        for i in range(self.nit):
            # eta = (i - 1) / (i + alpha)
            eta = i / (i + 1 + alpha)
            Z = Y + eta * (Y - Yold)
            Yold = Y
            w_new = w - A.T  @ Z @ B.T
            w_new[w_new < 0] = 0.0
            w_new = self.get_projection(w_new, self.Ad, index) # This line is modified by CHEN
            T = A @ w_new @ B
            [u,s,v] = np.linalg.svd(T)
            criterion = np.linalg.norm(w_new - w, ord='fro')
            constraint = np.linalg.norm(s[s > self.rho] - self.rho, ord=2)
            # print( 'iteration:', i+1, 'criterion: ', criterion, 'constraint: ', constraint)
            Yt = Z + gam * T
            [u1, s1, v1] = np.linalg.svd(Yt / gam, full_matrices=False)
            s1 = np.clip(s1, 0, self.rho)
            Y = Yt - gam * np.dot(u1 * s1, v1)
            if (criterion < self.criterion  and constraint < self.cnst):
                return w_new
        return w_new

    def No_Constraint(self, w, index_weight, Ad):

        w_new = self.get_projection(w, Ad, index_weight)

        return w_new
    
    def on_batch_end(self, batch, logs={}):  # def on_batch_end(self, batch, logs={}):
        # With Lipschitz Constraint
        if self.withConstraint:
            A = np.eye(self.model.layers[self.layers[-1]].output_shape[1])
            for index_weight in np.flip(self.layers):
                # index = self.layers.index(index_weight)
                B = np.eye(self.model.layers[self.layers[0]].input_shape[1]) # This line is modified by CHEN
                for index_weight_B in self.layers:
                    if (index_weight_B < index_weight):
                        B = np.transpose(self.model.layers[index_weight_B].get_weights()[0]) @ B
                w = np.transpose(self.model.layers[index_weight].get_weights()[0])
                wf = self.model.layers[index_weight].get_weights()
                if self.applyFista:
                    wf[0] = np.transpose(self.Constraint_Fista(w, A, B, index_weight))
                else:
                    wf[0] = np.transpose(self.Constraint(w, A, B, index_weight))
                self.model.layers[index_weight].set_weights(wf)
                A = A @ np.transpose(self.model.layers[index_weight].get_weights()[0])
        
        # Without Lipschitz Constraint
        else:
            for index_weight in self.layers:
                
                w = np.transpose(self.model.layers[index_weight].get_weights()[0])
                # print('w on batch end', w.shape)
                wf = self.model.layers[index_weight].get_weights()
                wf[0] = np.transpose(self.No_Constraint(w, index_weight, self.Ad))
                self.model.layers[index_weight].set_weights(wf)
        
        # # print weight matrix W
        # for i in self.layers:
        #     w_output_file = f"./weight_matrix/w_layer{i-1}_epoch{epoch}.out"
        #     w_mat = np.transpose(self.model.layers[i].get_weights()[0])
        #     np.savetxt(w_output_file, w_mat, delimiter='\t',fmt='%.1f')
    
        