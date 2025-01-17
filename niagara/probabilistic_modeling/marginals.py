import numpy as np
from scipy.stats import beta, rv_continuous, FitError
from scipy.optimize import minimize
from scipy.integrate import quad


class beta_couple:
    """ Implements a mixture of two beta distributions. """

    @staticmethod
    def scaled_beta_pdf(x, alpha, beta_param, p_min, p_max):
        return (1 / (p_max-p_min)) * beta.pdf( (x-p_min) / (p_max-p_min), alpha, beta_param)
    
    @staticmethod
    def scaled_beta_cdf(x, alpha, beta_param, p_min, p_max):
        return beta.cdf( (x-p_min) / (p_max-p_min), alpha, beta_param)
    
    @staticmethod
    def compute_scaled_beta_expectation(
        alpha, beta, p_min, p_max, func=lambda x: x, interval=[0,1]
    ):  
        a, b = interval
        result, _ = quad(
            lambda x: beta_couple.scaled_beta_pdf(x, alpha, beta, p_min, p_max) * func(x),
            a, 
            b
        )
        return result
    
    @staticmethod
    def compute_expectation(
        pi, alpha1, beta1, alpha2, beta2, p_min, p_max, func=lambda x: x, interval=[0,1]
    ):
        expectation = p_min + (p_max - p_min) * ( 
            pi * (alpha1 / (alpha1 + beta1)) + (1-pi) * (alpha2 / (alpha2 + beta2)) 
        )
        return expectation
        # return (
        #     pi * beta_couple.compute_scaled_beta_expectation(alpha1, beta1, p_min, p_max, func=func, interval=interval)
        #     + (1-pi) * beta_couple.compute_scaled_beta_expectation(alpha2, beta2, p_min, p_max, func=func, interval=interval)
        # )

    @staticmethod
    def scaled_beta_logpdf(x, alpha, beta_param, p_min, p_max):
        return -np.log(p_max-p_min) + beta.logpdf( (x-p_min) / (p_max-p_min), alpha, beta_param)

    @staticmethod
    def initialize_parameters(pi=0.5, alpha1=2.0, beta1=5.0, alpha2=5.0, beta2=2.0):
        return pi, alpha1, beta1, alpha2, beta2

    @staticmethod
    def estimate_scaled_beta_parameters(
            data, weights, alpha_init, beta_init, p_min, p_max, reg=0.0, maxiter=1000, tol=1e-6, show_warnings=False
        ):
        # Negative log-likelihood function
        def nll(params):
            alpha, beta_param = params
            if alpha <= 0 or beta_param <= 0:
                return np.inf
            # Log-PDF for scaled Beta distribution
            ll = weights * beta_couple.scaled_beta_logpdf(data, alpha, beta_param, p_min, p_max)
            return -np.sum(ll) + reg*(alpha**2 + beta_param**2)

        # Initial parameter estimates
        params_init = [alpha_init, beta_init]

        # Bounds to ensure parameters are positive
        bounds = [(1e-5, None), (1e-5, None)]

        # Minimize the negative log-likelihood
        result = minimize(nll, params_init, bounds=bounds, options={"maxiter": maxiter}, tol=tol)

        alpha_est, beta_est = result.x

        if (not result.success) and show_warnings:
            print("Numerical warning!")
            
        # if result.success:
        #     alpha_est, beta_est = result.x
        # else:
        #     raise FitError("Failed to estimate parameters for beta mixture")

        return alpha_est, beta_est

    @staticmethod
    def fit_mixture_scaled_beta(data, p_min, p_max, reg=0.0, responsibilities=None, max_iter=10000, tol=1e-6):
        # Initialize parameters
        pi, alpha1, beta1, alpha2, beta2 = beta_couple.initialize_parameters()
        n = len(data)

        log_likelihoods = []

        if responsibilities is None:
            responsibilities = np.ones_like(data)
        else:
            assert len(responsibilities) == len(data), "responsibilities must have same length as data"

        for iteration in range(max_iter):
            # E-step: Compute responsibilities
            p1 = pi * beta_couple.scaled_beta_pdf(data, alpha1, beta1, p_min, p_max)
            p2 = (1 - pi) * beta_couple.scaled_beta_pdf(data, alpha2, beta2, p_min, p_max)
            total_p = p1 + p2

            # Avoid division by zero
            total_p = np.maximum(total_p, 1e-6)

            # Responsibilities
            r1 = p1 / total_p
            r2 = p2 / total_p

            # M-step: Update parameters
            # Update mixing coefficient
            pi_new = np.mean(r1)

            # Update parameters for the first Beta distribution
            alpha1_new, beta1_new = beta_couple.estimate_scaled_beta_parameters(data, responsibilities*r1, alpha1, beta1, p_min, p_max, reg=reg, maxiter=max_iter, tol=tol)

            # Update parameters for the second Beta distribution
            alpha2_new, beta2_new = beta_couple.estimate_scaled_beta_parameters(data, responsibilities*r2, alpha2, beta2, p_min, p_max, reg=reg, maxiter=max_iter, tol=tol)

            # Compute log-likelihood
            log_likelihood = np.sum(np.log(total_p))
            log_likelihoods.append(log_likelihood)

            # Check for convergence
            if iteration > 0 and abs(log_likelihood - log_likelihoods[-2]) < tol:
                print(f"Converged at iteration {iteration}")
                break

            # Update parameters for next iteration
            pi = pi_new
            alpha1, beta1 = alpha1_new, beta1_new
            alpha2, beta2 = alpha2_new, beta2_new

        params = {
            'pi': pi,
            'alpha1': alpha1,
            'beta1': beta1,
            'alpha2': alpha2,
            'beta2': beta2,
            'log_likelihood': log_likelihood
        }

        return params
    
    @staticmethod
    def rvs(pi, alpha1, beta1, alpha2, beta2, p_min, p_max, size=1):
        beta1_samples = p_min + ( (p_max-p_min) * beta.rvs(a=alpha1, b=beta1,size=size) )
        beta2_samples = p_min + ( (p_max-p_min) * beta.rvs(a=alpha2, b=beta2,size=size) )
        component_choices = np.random.choice([0,1], size=size, p=[pi, 1-pi])
        return np.where(component_choices == 0, beta1_samples, beta2_samples)

    @staticmethod
    def pdf(x, pi, alpha1, beta1, alpha2, beta2, p_min, p_max):
        return (
            pi*beta_couple.scaled_beta_pdf(x, alpha1, beta1, p_min, p_max) + 
            (1-pi)*beta_couple.scaled_beta_pdf(x, alpha2, beta2, p_min, p_max)
        )
    
    @staticmethod
    def cdf(x, pi, alpha1, beta1, alpha2, beta2, p_min, p_max):
        return (
            pi*beta_couple.scaled_beta_cdf(x, alpha1, beta1, p_min, p_max) + 
            (1-pi)*beta_couple.scaled_beta_cdf(x, alpha2, beta2, p_min, p_max)
        )
    
### Implement instance-based version of beta couple

class betamix(rv_continuous):
    def __init__(self, p_min, p_max, pi, alpha1, beta1, alpha2, beta2):
        super().__init__(name=f"{self.__class__.__name__}")
        self.p_min = p_min
        self.p_max = p_max
        self.pi = pi
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.alpha2 = alpha2
        self.beta2 = beta2

    @staticmethod
    def fit(data, p_min, p_max, reg=0.0, tol=1e-12):
        if isinstance(data, list):
            data = np.array(data)
        assert isinstance(data, np.ndarray), "data must be a numpy array"
        assert data.ndim == 1, "data must be a 1D array"

        params = beta_couple.fit_mixture_scaled_beta(
            data, p_min, p_max, reg=reg
        )
        return (
            p_min, p_max, 
            params['pi'], params['alpha1'], params['beta1'], params['alpha2'], params['beta2']
        )

    def _cdf(self, x):
        return beta_couple.cdf(
            x, self.pi, self.alpha1, self.beta1, self.alpha2, self.beta2, self.p_min, self.p_max
        )
    
    def _pdf(self, x):
        return beta_couple.pdf(
            x, self.pi, self.alpha1, self.beta1, self.alpha2, self.beta2, self.p_min, self.p_max
        )
    
    def _rvs(self, size=1, *args, **kwargs):
        return beta_couple.rvs(
            pi=self.pi,
            alpha1=self.alpha1,
            beta1=self.beta1,
            alpha2=self.alpha2,
            beta2=self.beta2,
            p_min=self.p_min,
            p_max=self.p_max,
            size=size
        )


### Implement this marginal distribution!

class lumpy_betamix(rv_continuous):
    def __init__(self, p_min, p_max, w_min, w_max, pi, alpha1, beta1, alpha2, beta2):
        super().__init__(name=f"{self.__class__.__name__}")
        self.p_min = p_min
        self.p_max = p_max
        self.w_min = w_min
        self.w_max = w_max
        self.pi = pi
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.alpha2 = alpha2
        self.beta2 = beta2

    @staticmethod
    def fit(data, betamix_reg=0.05, p_min=None, p_max=None, tol=1e-12):
        """
        Fit this Lumpy BetaMix distribution to data.

        Parameters:
        data : numpy.ndarray
            The data to fit the distribution to.
        betamix_reg : float
            l2-Regularization parameter for fitting the beta mixture.
        p_min : float
            The minimum value in the data. If None, it is set to the minimum value in the data.
        p_max : float
            The maximum value in the data. If None, it is set to the maximum value in the data.
        """
        if isinstance(data, list):
            data = np.array(data)
        assert isinstance(data, np.ndarray), "data must be a numpy array"
        assert data.ndim == 1, "data must be a 1D array"

        if p_min is None:
            p_min = np.min(data)
        if p_max is None:
            p_max = np.max(data)

        # Fit the beta mixture on the data without discrete masses
        data_without_discrete_lumps = data[(data > p_min) & (data < p_max)]
        bm_params_est = beta_couple.fit_mixture_scaled_beta(
            data_without_discrete_lumps, p_min, p_max, reg=betamix_reg
        )
        # Estimate the discrete masses
        w_min_est = np.mean(data == p_min)
        w_max_est = np.mean(data == p_max)

        return (
            p_min, p_max, w_min_est, w_max_est, 
            bm_params_est['pi'], 
            bm_params_est['alpha1'], bm_params_est['beta1'],
            bm_params_est['alpha2'], bm_params_est['beta2']
        )
    
    def _rvs(self, size=1, *args, **kwargs):
        bm_samples = beta_couple.rvs(
            pi=self.pi,
            alpha1=self.alpha1,
            beta1=self.beta1,
            alpha2=self.alpha2,
            beta2=self.beta2,
            p_min=self.p_min,
            p_max=self.p_max,
            size=size
        )
        # components include mass at 0, beta mixture, mass at p_max
        component_choices = np.random.choice(
            a=[0,1,2],
            size=size,
            p=[self.w_min, 1 - self.w_min-self.w_max, self.w_max],
        )
        return np.where(component_choices == 1, bm_samples,
                 np.where(component_choices == 2, self.p_max, self.p_min)
        )

    def _cdf(self, x):
        discrete_part_low = np.where(x >= self.p_min, self.w_min, 0)
        discrete_part_high = np.where(x >= self.p_max, self.w_max, 0)

        w = self.w_min + self.w_max
        continuous_part_cdf = (1 - w) * beta_couple.cdf(
            x, self.pi, self.alpha1, self.beta1, self.alpha2, self.beta2, self.p_min, self.p_max
        )
        return discrete_part_low + continuous_part_cdf + discrete_part_high
    
    def compute_expectation(self, func = lambda x: x, interval = [0, 1]):
        """ Compute the expectation against a function on an interval. """
        a, b = interval
        lumpy_high = self.w_max * func(self.p_max) * (self.p_max >= a) * (self.p_max < b)
        lumpy_low = self.w_min * func(self.p_min) * (self.p_min >= a) * (self.p_min < b)
        cts_contribution = (1 - self.w_max - self.w_min) * beta_couple.compute_expectation(
            pi=self.pi,
            alpha1=self.alpha1,
            beta1=self.beta1,
            alpha2=self.alpha2,
            beta2=self.beta2,
            p_min=self.p_min,
            p_max=self.p_max,
            func=func,
            interval=[a,b]
        )
        return lumpy_low + lumpy_high + cts_contribution
