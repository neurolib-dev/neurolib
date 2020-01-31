# This file contains helper functions for DEAP
# including additional mutation, crossover and selection operators.

import random
import numpy as np


def indivAsDict_adapt(individual, ParametersInterval, paramInterval):
    """
    Convert an individual to a dictionary
    """
    return ParametersInterval(*(individual[: len(paramInterval)]))._asdict().copy()


def generate_random_pars_adapt(paramInterval):
    """
    Generate a sequence of random parameters from a ParamsInterval using a uniform distribution.
    Format: [mean_par1, mean_par2, ..., sigma_par1, sigma_par2, ...]
    The second half of the parameter list is set of adaptive mutation std deviation parameters.
    """
    params = list(map(lambda pI: np.random.uniform(pI[0], pI[1]), paramInterval))

    # The innitial adaptation parameters are chosen according to the inidial parameter range:
    defaultAdaptation = list(map(lambda pI: (abs(pI[1] - pI[0]) / 3), paramInterval))
    # add sigma's to the list of means
    params.extend(defaultAdaptation)
    return params


def check_param_validity(individual, paramInterval):
    """
    Check if an individual is within the specified bounds
    Return True if it is correct, False otherwise
    """
    for i, v in enumerate(paramInterval):
        if individual[i] < v[0] or individual[i] > v[1]:
            return False
    return True


### Selection operators ###
# Rank selection
def selRank(individuals, k, s=1.5):
    """
    Select k individual from a population using the Rank selection
    The individual are selected according to the fitness rank
    In case of multiobjective fitness function, the weighted sum of fitness objective will be used.

        n the rank selection, individual are selected with a probability depending on their rank.
    """
    # Sort individual according to their rank, the first indiv in the list is the one with the best fitness
    s_inds = sorted(individuals, key=lambda iv: np.nansum(iv.fitness.wvalues), reverse=True)

    mu = len(individuals)

    # Probability of drawing individuals i in s_inds
    P_indiv = ((2 - s) / mu + 2 * (s - 1) / (mu * (mu - 1)) * np.arange(mu)).tolist()
    P_indiv.reverse()

    sum_P = sum(P_indiv)

    chosen = []
    for i in range(k):
        u = random.random() * sum_P
        sum_ = 0
        for i, ind in enumerate(s_inds):
            sum_ += P_indiv[i]
            if sum_ > u:
                chosen.append(ind)
                break
    return chosen


# Select best
def selBest_multiObj(individuals, k):
    """
    Select the best k individuals.

    This function accept multiobjective function by summing the fitness all of objectives.
    """
    # Sort individual according to their rank, the first indiv in the list is the one with the best fitness
    return sorted(individuals, key=lambda iv: np.nansum(iv.fitness.wvalues), reverse=True)[:k]


### Crossover operators ###

# This crossover was taken from DEAP but modified to
#   - add a boolean return value giving information on if a crossover happenned
#   - switch the adaptive mutation rate too
def cxUniform_adapt(ind1, ind2, indpb):
    """Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals. The attributes are swapped according to the
    *indpb* probability.
    The individuals are composed of the gene values first and then the mutation rates.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param indpb: Independent probabily for each attribute to be exchanged.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    for i in range(size // 2):
        if random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]
            iAdapt = i + size // 2
            ind1[iAdapt], ind2[iAdapt] = ind2[iAdapt], ind1[iAdapt]

    return ind1, ind2


def cxUniform_normDraw(ind1, ind2, indpb):
    """Executes a uniform crossover that modify in place the two individuals.
    The attributes of the 2 individuals are set according to a normal distribution whose mean is
    the mean between both individual attributes and the standard deviation the distance between the 2 attributes.
    The individuals are composed of the gene values first and then the mutation rates.

    Warning: a check should be done afterward on the parameter to be sure they are not out of bound.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param indpb: Independent probabily for each attribute to be exchanged.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    for i in range(size // 2):
        if random.random() < indpb:
            mu = np.mean([ind1[i], ind2[i]])
            sigma = np.abs(ind1[i] - ind2[i])
            ind1[i] = random.normalvariate(mu, sigma)
            ind2[i] = random.normalvariate(mu, sigma)

    return ind1, ind2


def cxUniform_normDraw_adapt(ind1, ind2, indpb):
    """Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals.
    The attributes of the 2 individuals are set according to a normal distribution whose mean is
    the mean between both individual attributes and the standard deviation the distance between the 2 attributes.
    The individuals are composed of the gene values first and then the mutation rates.

    Warning: a check should be done afterward on the parameter to be sure they are not out of bound.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param indpb: Independent probabily for each attribute to be exchanged.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    for i in range(size // 2):
        if random.random() < indpb:
            mu = np.mean([ind1[i], ind2[i]])
            sigma = np.abs(ind1[i] - ind2[i])
            ind1[i] = random.normalvariate(mu, sigma)  # in-place modification!
            ind2[i] = random.normalvariate(mu, sigma)  # in-place modification!
            iAdapt = i + size // 2  # adaptive parameters, start at half of the list
            ind1[iAdapt], ind2[iAdapt] = ind2[iAdapt], ind1[iAdapt]

    return ind1, ind2


### Mutation operators ###

# Adaptive mutation with m different stepsizes
def adaptiveMutation_nStepSize(mutant, gamma_gl=None, gammas=None):
    """
    Perform an uncorrelated adaptive mutation with n step sizes on the mutant

    Warning: the mutations is in place, i.e. it modifies the given individual
    Parameters:
        :param mutant:      Inidivual to mutate. This should a sequence of length 2 * n_params 
                                ( the last n_params element being the individual adaptation rates)
        :param gamma_gl:   Global adaptive mutation param ( should be proportional to 1/sqrt(2 n_params ) )
        :param gammas:      Adaptive mutation parameters ( should be proportional to 1/sqrt(2 sqrt(n_params) ) )

    :returns: the mutant

    """
    nParams = len(mutant) // 2
    oldParams = mutant[0:nParams]
    oldSigmas = mutant[nParams:]

    if gamma_gl is None:
        gamma_gl = 1 / np.sqrt(2 * nParams)

    if gammas is None:
        gammas = [1 / np.sqrt(2 * np.sqrt(nParams))] * nParams

    newSigmas = [
        oldSigmas[i] * np.exp(gammas[i] * np.random.randn() + gamma_gl * np.random.randn()) for i in range(nParams)
    ]
    newParams = [oldParams[i] + newSigmas[i] * np.random.randn() for i in range(nParams)]

    mutant[:] = newParams + newSigmas

    return mutant
