import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gaussian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            k_mean = KMeans(self.n_cluster,max_iter=self.max_iter, e=self.e)
            self.means, classes, _ = k_mean.fit(x)
            unique, counts = np.unique(classes, return_counts=True)

            self.pi_k = counts / counts.sum()
            self.variances = np.zeros((len(unique),D,D))
            for c,mu in zip(unique, self.means):
                mask = (classes == c)
                x_mu = x - self.means[c]
                x_mu[~mask] = 0
                self.variances[c] = np.dot(x_mu.T,x_mu) / counts[c]

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform
            
            self.means = np.array([[np.random.rand() for _ in range(D)] for _ in range(self.n_cluster)])
            self.pi_k = np.array([1/self.n_cluster for _ in range(self.n_cluster)])
            self.variances = np.array([np.eye(D,D) for _ in range(self.n_cluster)])

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # means, variances, pi_k = self.means, self.variances, self.pi_k
        # # print(self.means, self.variances, self.pi_k)
        # def compute_membership(self, x, means, variances, pi_k):
        #      gaussians = [self.Gaussian_pdf(means[i], variances[i])
        #                   for i in range(self.n_cluster)]
 
        #      N, D = x.shape
        #      membership = np.zeros((N, self.n_cluster))
        #      for i in range(N):
        #          for j in range(self.n_cluster):
        #              membership[i][j] = pi_k[j]*gaussians[j].getLikelihood(x[i])
        #      return membership/np.sum(membership, axis=1).reshape([-1, 1])
 
        # l = self.compute_log_likelihood(x, means, variances, pi_k)

        # for j in range(self.max_iter):
        #     membership = compute_membership(self, x, means, variances, pi_k)

        #     # recompute mean
        #     for i in range(self.n_cluster):
        #         t = membership[:, i].reshape([-1, 1])
        #         means[i] = np.sum(t*x, axis=0) / (np.sum(t)+1e-10)
        #         variances[i] = (t * (x-means[i])).T @ (x -
        #                                                means[i]) / (np.sum(t) + 1e-10)

        #     pi_k = np.sum(membership, axis=0)/N
        #     l_new = self.compute_log_likelihood(x, means, variances, pi_k)
            
        #     # if j == 7:
        #     #     print(means, variances, pi_k)
        #     #     exit()

        #     if (np.abs(l_new-l) < self.e):
        #         self.means = means
        #         self.variances = variances
        #         self.pi_k = pi_k
        #         return j+1
        #     l = l_new

        
        # self.means = means
        # self.variances = variances
        # self.pi_k = pi_k
        # return self.max_iter


        ll = float('inf')
        for i in range(self.max_iter):
            new_ll = self.compute_log_likelihood(x,self.means, self.variances, self.pi_k)
            Nk = np.sum(self.gammas, axis=0)

            for c in range(self.n_cluster):
                self.means[c] = np.dot(self.gammas[:,c], x) / Nk[c]
                x_mu = x - self.means[c]
                # self.variances[c] = (self.gammas[:,c].reshape([-1,1]) * (x_mu)).T @ x_mu / (Nk[c] + 1e-10)
                self.variances[c] = np.dot(np.multiply(self.gammas[:,c],x_mu.T), x_mu) / (Nk[c]+ 1e-10)
                self.pi_k[c] = Nk[c] / x.shape[0]

            if np.abs(ll-new_ll) < self.e:
                return i+1

            ll = new_ll

        return i

		
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        D = self.means.shape[1]
        samples = np.random.standard_normal(size=(N, D))
        for i in range(N):
            component = np.random.choice(self.n_cluster, p=self.pi_k)
            samples[i] = np.random.multivariate_normal(mean=self.means[component], cov=self.variances[component])

        return samples        

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k    
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)

        self.gammas = np.zeros((x.shape[0],self.n_cluster))
        log_likelihood = 0
        gauss_pdfs = [self.Gaussian_pdf(means[j], variances[j]) for j in range(self.n_cluster)]

        for i in range(x.shape[0]):
            p = 0
            for j in range(self.n_cluster):
                self.gammas[i,j] = pi_k[j] * gauss_pdfs[j].getLikelihood(x[i])
                p += self.gammas[i,j]
            log_likelihood += np.log(p)
        
        gamma_sum = np.sum(self.gammas, axis=1)
        self.gammas /= gamma_sum[:,None]

        return float(log_likelihood)

    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit

            while np.linalg.matrix_rank(self.variance) != len(self.variance):
                self.variance += np.eye(len(self.variance)) * 1e-3
            self.inv = np.linalg.inv(self.variance)
            self.c = (2 * np.pi) ** len(self.variance) * np.linalg.det(self.variance)


        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)')/sqrt(c)
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # likelihood = np.exp(-0.5 * np.dot(np.dot((x-self.mean),self.inv),(x-self.mean).T)) / np.sqrt(self.c)


            p = np.exp(-0.5* np.dot(np.dot((x-self.mean),self.inv),(x-self.mean).T))/np.sqrt(self.c)
            
            return p
