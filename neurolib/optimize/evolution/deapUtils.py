# This file contains helper functions for DEAP
# including additional mutation, crossover and selection operators.

import random
import copy
import numpy as np


# def indivAsDict_adapt(individual, ParametersInterval, paramInterval):
#     """
#     Convert an individual to a dictionary
#     """
#     return ParametersInterval(*(individual[: len(paramInterval)]))._asdict().copy()


def randomParameters(paramInterval):
    """
    Generate a sequence of random parameters from a ParamsInterval using a uniform distribution.
    Format: [mean_par1, mean_par2, ...]
    """
    params = [np.random.uniform(*pI) for pI in paramInterval]
    return params


def randomParametersAdaptive(paramInterval):
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


def mutateUntilValid(pop, paramInterval, toolbox, MUTATE_P={}, maxTries=100):
    """Checks the validity of new individuals' parameter. If they are invalid 
    (for example if they are out of the predefined paramter space bounds), 
    mutate the individual, until valid.

    :param pop: population to mutate
    :param paramInterval: parameter interval (from parameterSpace.named_tuple)
    :param toolbox: deap toolbox
    :param maxTries: how many mutations to try until valid
    """
    # mutate individuald until valid, max 100 times
    for i, ind in enumerate(pop):

        ind_bak = copy.copy(ind)
        toolbox.mutate(pop[i], **MUTATE_P)

        nMutations = 0
        while not checkParamValidity(pop[i], paramInterval) and nMutations < maxTries:
            pop[i] = copy.copy(ind_bak)
            toolbox.mutate(pop[i])
            nMutations += 1

        # if it didn't work, set the individual to the boundary
        for l, v in enumerate(paramInterval):
            if pop[i][l] < v[0]:
                pop[i][l] = float(v[0])
            elif pop[i][l] > v[1]:
                pop[i][l] = float(v[1])


def checkParamValidity(individual, paramInterval):
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
def selRank(pop, k, s=1.5):
    """
    Select k individuals from a population using rank selection. (Eiben&Smith, p.81)
    Individuals are selected according to the fitness rank.
    To support multiobjective fitness functions, the weighted sum of fitness is used.

    :param pop: population
    :type pop: list
    :param k: number of individuals to select
    :type k: int
    :param s: selection probability parameter
    :type s: float

    :return: population of selected individuals
    :rtype: list
    """
    # Sort individual according to their rank, the first indiv in the list is the one with the best fitness
    s_inds = sorted(pop, key=lambda iv: np.nansum(iv.fitness.wvalues), reverse=True)

    mu = len(pop)

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


# # Wheel selection
# # This code is not compatible with multiobjective fitness functions! Use np.nansum(iv.fitness.wvalues) instead!
# def selWheel(individuals,k):
#     '''
#     Select k individual from a population using the Roulette selection
#     Since we are trying to minimize the distance, we use the inverse fitness function as a probability

#     This code is inspired from DEAP.toolbox.selRoulette
#     '''
#     s_inds = sorted(individuals, key=attrgetter("fitness"), reverse=True)
#     sum_invfits = sum(1/ind.fitness.values[0] for ind in individuals)

#     chosen = []
#     for i in range(k):
#         u = random.random() * sum_invfits
#         sum_ = 0
#         for ind in s_inds:
#             sum_ += 1 / ind.fitness.values[0]
#             if sum_ > u:
#                 chosen.append(ind)
#                 break
#     return chosen


# Select best
def selBest_multiObj(pop, k):
    """
    Select the best k individuals.

    This function accept multiobjective function by summing the fitness all of objectives.
    """
    # Sort individual according to their rank, the first indiv in the list is the one with the best fitness
    return sorted(pop, key=lambda iv: np.nansum(iv.fitness.wvalues), reverse=True)[:k]


# ### Crossover operators ###

# # This crossover was taken from DEAP but modified to
# #   - add a boolean return value giving information on if a crossover happenned
# #   - switch the adaptive mutation rate too
# def cxUniform_adapt(ind1, ind2, indpb):
#     """Executes a uniform crossover that modify in place the two
#     :term:`sequence` individuals. The attributes are swapped according to the
#     *indpb* probability.
#     The individuals are composed of the gene values first and then the mutation rates.

#     :param ind1: The first individual participating in the crossover.
#     :param ind2: The second individual participating in the crossover.
#     :param indpb: Independent probabily for each attribute to be exchanged.
#     :returns: A tuple of two individuals.

#     This function uses the :func:`~random.random` function from the python base
#     :mod:`random` module.
#     """
#     size = min(len(ind1), len(ind2))
#     for i in range(size // 2):
#         if random.random() < indpb:
#             ind1[i], ind2[i] = ind2[i], ind1[i]
#             iAdapt = i + size // 2
#             ind1[iAdapt], ind2[iAdapt] = ind2[iAdapt], ind1[iAdapt]

#     return ind1, ind2


# def cxUniform_normDraw(ind1, ind2, indpb):
#     """Executes a uniform crossover that modify in place the two individuals.
#     The attributes of the 2 individuals are set according to a normal distribution whose mean is
#     the mean between both individual attributes and the standard deviation the distance between the 2 attributes.
#     The individuals are composed of the gene values first and then the mutation rates.

#     Warning: a check should be done afterward on the parameter to be sure they are not out of bound.

#     :param ind1: The first individual participating in the crossover.
#     :param ind2: The second individual participating in the crossover.
#     :param indpb: Independent probabily for each attribute to be exchanged.
#     :returns: A tuple of two individuals.

#     This function uses the :func:`~random.random` function from the python base
#     :mod:`random` module.
#     """
#     size = min(len(ind1), len(ind2))
#     for i in range(size // 2):
#         if random.random() < indpb:
#             mu = np.mean([ind1[i], ind2[i]])
#             sigma = np.abs(ind1[i] - ind2[i])
#             ind1[i] = random.normalvariate(mu, sigma)
#             ind2[i] = random.normalvariate(mu, sigma)

#     return ind1, ind2


def cxNormDraw_adapt(ind1, ind2, sigma_scale=2.0):
    """The new attributes of the two individuals are set according to a normal distribution whose mean is
    the mean between both individual's attributes and the standard deviation being the distance between the two attributes.
    
    Similar to mutation parameter described in Ono et al 2003 but with only 2 parents (and not 3).

    Info: The individuals are composed of the gene values first and then the mutation rates.
    Warning: a check should be done afterward on the parameter to be sure they are not out of bound.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param sigma_scale: Scaling of sigma (distance of parents / sigma_scale)
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    for i in range(size // 2):
        mu = float(np.mean([ind1[i], ind2[i]]))
        sigma = float(np.abs(ind1[i] - ind2[i])) / sigma_scale
        # In Ono 2003, they draw only one random number r and
        # ind1 = mean - r * sigma
        # ind2 = mean + r * sigma
        # We draw two independent parameters here
        ind1[i] = random.gauss(mu, sigma)  # in-place modification!
        ind2[i] = random.gauss(mu, sigma)

        iAdapt = i + size // 2  # adaptive parameters, start at half of the list
        # ind1[iAdapt], ind2[iAdapt] = ind2[iAdapt], ind1[iAdapt]
        mu_adapt = float(np.mean([ind1[iAdapt], ind2[iAdapt]]))
        sigma_adapt = float(np.abs(ind1[iAdapt] - ind2[iAdapt])) / sigma_scale
        ind1[iAdapt] = random.gauss(mu_adapt, sigma_adapt)
        ind2[iAdapt] = random.gauss(mu_adapt, sigma_adapt)

    return ind1, ind2


def cxUniform_adapt(ind1, ind2, indpb):
    """The new attributes of the two individuals are set according to a normal distribution whose mean is
    the mean between both individual's attributes and the standard deviation being the distance between the two attributes.
    
    Info: The individuals are composed of the gene values first and then the mutation rates.
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
            ind1[i], ind2[i] = ind2[i], ind1[i]  # in-place modification!
            iAdapt = i + size // 2  # adaptive parameters, start at half of the list
            ind1[iAdapt], ind2[iAdapt] = ind2[iAdapt], ind1[iAdapt]

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
            mu = float(np.mean([ind1[i], ind2[i]]))
            sigma = float(np.abs(ind1[i] - ind2[i])) / 4
            ind1[i] = random.normalvariate(mu, sigma)  # in-place modification!
            ind2[i] = random.normalvariate(mu, sigma)  # in-place modification!
            iAdapt = i + size // 2  # adaptive parameters, start at half of the list
            ind1[iAdapt], ind2[iAdapt] = ind2[iAdapt], ind1[iAdapt]

    return ind1, ind2


### Mutation operators ###

# Adaptive mutation with m different stepsizes
def gaussianAdaptiveMutation_nStepSizes(individual, gamma_gl=None, gamma=None):
    """
    Perform an uncorrelated adaptive mutation with n step sizes on the individual

    Warning: the mutations is in place, i.e. it modifies the given individual
    Parameters:
        :param individual: Inidivual to mutate. This should a sequence of length 2 * n_params 
        the last n_params elements being the individual adaptation rates)
        :param gamma_gl: Global adaptive mutation param ( should be proportional to 1/sqrt(2 n_params ) )
        :param gamma: Adaptive mutation parameters ( should be proportional to 1/sqrt(2 sqrt(n_params) ) )

    :returns: the individual

    """
    n_params = len(individual) // 2
    oldParams = individual[0:n_params]
    oldSigmas = individual[n_params:]

    if gamma_gl is None:
        gamma_gl = 1 / np.sqrt(2 * n_params)

    if gamma is None:
        gamma = 1 / np.sqrt(2 * np.sqrt(n_params))

    randn_global = float(np.random.randn())

    newSigmas = [
        oldSigmas[i] * float(np.exp(gamma * float(np.random.randn()) + gamma_gl * randn_global))
        for i in range(n_params)
    ]
    newParams = [oldParams[i] + newSigmas[i] * float(np.random.randn()) for i in range(n_params)]

    individual[:] = newParams + newSigmas

    return (individual,)
