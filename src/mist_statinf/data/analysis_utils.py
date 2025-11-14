import numpy as np
from scipy.stats import kurtosis

"""
This module provides utilities to generate univariate datasets from various statistical distributions,
which is used in our plotting code and for testing purposes.
"""

global_prior = {
    'Gaussian': {'mean': (0, 1), 'std': (0.1, 2)},
    'Uniform': {'low': (-1, 0), 'high': (1, 2)},
    'Exponential': {'scale': (0.1, 2)},
    # 'Beta': {'a': (0.5, 5), 'b': (0.5, 5)},
    'Gamma': {'shape': (1, 5), 'scale': (0.5, 2)},
    'Log-Normal': {'mean': (0, 1), 'sigma': (0.1, 1)},
    'Triangular': {'left': (-1, 0), 'mode': (0.1, 1), 'right': (1, 2)},
    'Laplace': {'loc': (0, 1), 'scale': (0.5, 2)},
    # 'VonMises': {'mu': (0, np.pi), 'kappa': (0.5, 5)},
    # 'Arcsine': {'a': (0, 1), 'b': (1, 2)},
    # 'Bimodal': {'mean1': (-2, -1), 'std1': (0.5, 1), 'mean2': (1, 2), 'std2': (0.5, 1)},
    'StudentT': {'df': (2.001, 15)},
    'ChiSquare': {'df': (1, 10)},
    'Rayleigh': {'scale': (0.5, 2)},
    'Gumbel': {'loc': (0, 1), 'scale': (0.5, 2)},
    'Logistic': {'loc': (0, 1), 'scale': (0.5, 2)},
    # 'PowerLaw': {'a': (1.5, 3)},
    # 'SkewNormal': {'mean': (-1, 1), 'std': (0.5, 2), 'skewness': (2, 10)},
    # 'Erlang': {'shape': (2, 5), 'scale': (0.5, 2)}
}


def _generate_1d_dataset(n_row, distribution, params):
    if distribution == 'Gaussian':
        data = np.random.normal(params['mean'], params['std'], n_row)
        pop_std = params['std']
        excess_kurtosis = 0.
    elif distribution == 'Uniform':
        data = np.random.uniform(params['low'], params['high'], n_row)
        pop_std = (params['high'] - params['low']) / np.sqrt(12)
        excess_kurtosis = -6. / 5
    elif distribution == 'Exponential':
        data = np.random.exponential(params['scale'], n_row)
        pop_std = params['scale']
        excess_kurtosis = 6.
    # elif distribution == 'Beta':
    #     data = np.random.beta(params['a'], params['b'], n_row)
    #     pop_std = np.sqrt(
    #         (params['a'] * params['b']) / ((params['a'] + params['b']) ** 2 * (params['a'] + params['b'] + 1)))
    #     excess_kurtosis = (6 * (params['a'] - params['b']) ** 2 * (params['a'] + params['b'] + 1) - params['a'] *
    #                        params['b'] * (params['a'] + params['b'] + 2)) / (
    #                               params['a'] * params['b'] * (params['a'] + params['b'] + 2) * (
    #                               params['a'] + params['b'] + 3))
    elif distribution == 'Gamma':
        data = np.random.gamma(params['shape'], params['scale'], n_row)
        pop_std = np.sqrt(params['shape']) * params['scale']
        excess_kurtosis = 6 / params['shape']
    elif distribution == 'Log-Normal':
        data = np.random.lognormal(params['mean'], params['sigma'], n_row)
        pop_std = np.sqrt((np.exp(params['sigma'] ** 2) - 1) * np.exp(2 * params['mean'] + params['sigma'] ** 2))
        excess_kurtosis = np.exp(4 * params['sigma'] ** 2) + 2 * np.exp(3 * params['sigma'] ** 2) + 3 * np.exp(
            2 * params['sigma'] ** 2) - 6
    elif distribution == 'Triangular':
        data = np.random.triangular(params['left'], params['mode'], params['right'], n_row)
        pop_std = np.sqrt(params['left'] ** 2 + params['mode'] ** 2 + params['right'] ** 2 -
                          params['left'] * params['mode'] - params['left'] * params['right'] - params['mode'] * params[
                              'right']) / np.sqrt(18.)  # Complex formula
        excess_kurtosis = - 3. / 5  # Complex formula
    elif distribution == 'Laplace':
        data = np.random.laplace(params['loc'], params['scale'], n_row)
        pop_std = np.sqrt(2) * params['scale']
        excess_kurtosis = 3
    elif distribution == 'StudentT':
        data = np.random.standard_t(params['df'], n_row)
        pop_std = np.sqrt(params['df'] / (params['df'] - 2)) if params['df'] > 2 else np.inf
        excess_kurtosis = 6 / (params['df'] - 4) if params['df'] > 4 else np.inf
    elif distribution == 'ChiSquare':
        data = np.random.chisquare(params['df'], n_row)
        pop_std = np.sqrt(2 * params['df'])
        excess_kurtosis = 12 / params['df']
    elif distribution == 'Rayleigh':
        data = np.random.rayleigh(params['scale'], n_row)
        pop_std = params['scale'] * np.sqrt((4 - np.pi) / 2)
        excess_kurtosis = 0.245
    elif distribution == 'Gumbel':
        data = np.random.gumbel(params['loc'], params['scale'], n_row)
        pop_std = np.pi / np.sqrt(6) * params['scale']
        excess_kurtosis = 2.4
    elif distribution == 'Logistic':
        data = np.random.logistic(params['loc'], params['scale'], n_row)
        pop_std = params['scale'] * np.pi / np.sqrt(3)
        excess_kurtosis = 1.2
    else:
        raise ValueError("Unsupported distribution")

    return data.reshape(-1, 1), {'std': pop_std, 'excess_kurtosis': excess_kurtosis}


#### These are for testing purposes ####
def generate_toy_dataset(n_row, distribution, params_prior=None):
    if params_prior is None:
        params_prior = global_prior

    params = {k: np.random.uniform(*v) for k, v in params_prior[distribution].items()}
    dataset, extra_infos = _generate_1d_dataset(n_row, distribution, params)
    # return {'dataset': standardize_dataset(dataset), 'parameters': params, 'extra_infos': extra_infos}
    return {'dataset': dataset, 'parameters': params, 'extra_infos': extra_infos}


def validate_empirical_moments(dataset_output, distribution):
    sample_size = 5000
    extra_infos = dataset_output['extra_infos']
    dataset, _ = _generate_1d_dataset(sample_size, distribution, dataset_output['parameters'])

    emp_std = np.std(dataset)
    emp_excess_kurtosis = kurtosis(dataset, fisher=True)

    print(emp_std, extra_infos['std'])
    # print(emp_excess_kurtosis, extra_infos['excess_kurtosis'])
    assert np.allclose(emp_std, extra_infos['std'], atol=1e-2), "Empirical std does not match population std"
    assert np.allclose(emp_excess_kurtosis, extra_infos['excess_kurtosis'],
                       atol=3e-1), "Empirical excess kurtosis does not match population kurtosis"


if __name__ == "__main__":
    for distrib in global_prior.keys():
        print(f"distrib: {distrib}")
        dataset = generate_toy_dataset(100, distrib, params_prior=global_prior)
        validate_empirical_moments(dataset, distrib)