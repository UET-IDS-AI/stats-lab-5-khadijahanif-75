import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(scale=1, size=n)
    count = np.sum((samples > a) & (samples < b))
    return count / n


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    return (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-((x-mu)**2)/(2*sigma**2))


def posterior_probability(time):
    # priors
    pA = 0.3
    pB = 0.7

    likeA = np.exp(-(time-40)**2 / 4)
    likeB = np.exp(-(time-45)**2 / 4)

    numerator = pB * likeB
    denominator = pA * likeA + numerator

    return numerator / denominator


def simulate_posterior_probability(time, n=100000):
    # sample classes according to priors
    classes = np.random.choice(['A','B'], size=n, p=[0.3,0.7])

    samples = []

    for c in classes:
        if c == 'A':
            samples.append(np.random.normal(40,2))
        else:
            samples.append(np.random.normal(45,2))

    samples = np.array(samples)

    # select samples close to observed time
    mask = np.abs(samples - time) < 0.5

    if np.sum(mask) == 0:
        return 0

    selected_classes = classes[mask]

    return np.sum(selected_classes == 'B') / len(selected_classes)
