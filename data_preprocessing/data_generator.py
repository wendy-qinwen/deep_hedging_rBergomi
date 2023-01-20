# ## Generate Heston data
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

tqdm.pandas()
import QuantLib as ql
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_lets_be_rational.exceptions import BelowIntrinsicException
import sklearn.utils
import logging

import numpy as np
from matplotlib import pyplot as plt
from rbergomi.rbergomi import rBergomi
from scipy.stats import truncnorm

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def heston_pricer(lambd, vbar, eta, rho, v0, r, q, tau, S0, K):
    """Computes European Call price under Heston dynamics with closedform solution.
    
    Parameters:
    -----------
        lambd: mean-reversion speed
        vbar: long-term average variance
        eta: volatility of variance
        rho: correlation between stock and vol
        v0: spot variance
        r: risk-free interest rate
        q: dividend rate
        tau: time to maturity in years (365 trading days per year)
        S0: initial spot price
        K: strike price
    """
    today = datetime.date.today()
    ql_date = ql.Date(today.day, today.month, today.year)
    day_count = ql.Actual365Fixed()
    ql.Settings.instance().evaluationDate = ql_date

    # option data
    option_type = ql.Option.Call
    payoff = ql.PlainVanillaPayoff(option_type, K)
    maturity_date = ql_date + int(round(tau * 365))
    exercise = ql.EuropeanExercise(maturity_date)
    european_option = ql.VanillaOption(payoff, exercise)

    # Heston process
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, r, day_count))
    dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, q, day_count))
    heston_process = ql.HestonProcess(flat_ts, dividend_yield, spot_handle, v0, lambd, vbar, eta, rho)

    engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process), 1e-15, int(1e6))
    european_option.setPricingEngine(engine)

    # check numerical stability
    try:
        price = european_option.NPV()
        if price <= 0 or price + K < S0:
            iv = np.nan
            logging.debug("NumStabProblem: Price {}. Intrinsic {}. Time {}. Strike {}.".format(price, S0 - K, tau, K))
        else:
            logging.debug("Success: Price {} > intrinsic {}".format(price, S0 - K))
            iv = implied_volatility(price, S0, K, tau, r, 'c')
    except RuntimeError:
        logging.info("RuntimeError: Intrinsic {}. Time {}. Strike {}.".format(S0 - K, tau, K))
        price = np.nan
        iv = np.nan
    return price, iv


def rBergomi_pricer(H, eta, rho, v0, tau, K, S0, MC_samples=40000):
    """Computes European Call price under rBergomi dynamics with MC sampling.

    Parameters:
    -----------
        H: Hurst parameter
        eta: volatility of variance
        rho: correlation between stock and vol
        v0: spot variance
        tau: time to maturity in years (365 trading days per year)
        K: strike price
    """
    try:
        rB = rBergomi(n=365, N=MC_samples, T=tau, a=H - 0.5)
        dW1, dW2 = rB.dW1(), rB.dW2()
        Y = rB.Y(dW1)
        dB = rB.dB(dW1, dW2, rho)
        xi = v0
        V = rB.V(Y, xi, eta)
        S = rB.S(V, dB)
        ST = S[:, -1]
        price = np.mean(np.maximum(ST - K, 0))
    except:
        return np.nan, np.nan

    # check numerical stability
    if price <= 0 or price + K < S0:
        iv = np.nan
        logging.debug("NumStabProblem: Price {}. Intrinsic {}. Time {}. Strike {}.".format(price, S0 - K, tau, K))
    else:
        logging.debug("Success: Price {} > intrinsic {}".format(price, S0 - K))
        iv = implied_volatility(price, S0, K, tau, 0, 'c')
    return price, iv


def param_generator(H_generator=truncnorm(-1.2, 8.6, 0.07, 0.05),
                    eta_generator=truncnorm(-3, 3, 2.5, 0.5),
                    rho_generator=truncnorm(-0.25, 2.25, -0.95, 0.2),
                    v0_generator=truncnorm(-2.5, 7, 0.3, 0.1)):
    rslt = {
        'H': H_generator.rvs(),
        'eta': eta_generator.rvs(),
        'rho': rho_generator.rvs(),
        'v0': v0_generator.rvs() ** 2
    }
    return rslt


def generate_rBergomi_sample(K, T, param_generator, S0=1.0):
    """ Generates a rBergomi sample with random parameters
    """
    counter = 0
    while counter < 10:
        params = param_generator()
        H, eta, rho, v0 = params['H'], params['eta'], params['rho'], params['v0']
        _, iv = rBergomi_pricer(H, eta, rho, v0, T, K, S0)
        if np.isnan(iv):
            counter += 1
        else:
            break
    else:
        logging.warning("Tried 10 times, none valid sample obtained.")
    sample = {
        'H': H,
        'eta': eta,
        'rho': rho,
        'v0': v0,
        'iv': iv
    }
    return sample


# Generate labeled data

if __name__ == '__main__':
    # Load previously generated $(m, T)$ pairs

    K_T = pd.read_csv("./data/strike_maturity.csv", index_col=0)
    # Initialize the data frame to store labeled data
    columns = ['lambda', 'vbar', 'eta', 'rho', 'v0', 'iv']
    data_nn = pd.DataFrame(index=K_T.index, columns=columns)
    data_nn = pd.concat([K_T, data_nn], axis=1)
    # Specify model parameters' distribution. For Heston model, we draw the model parameter
    # $\mu = (\eta, \rho, \lambda, \bar{v}, v_0)$ from uniform distribution as proposed in the Table 1 of the paper.

    # PARAMETERS
    n_samples = K_T.shape[0]
    # Heston parameter, bounds by Moodley (2005)
    lambd_bounds = [0, 10]
    vbar_bounds = [0, 1]
    eta_bounds = [0, 5]
    rho_bounds = [-1, 0]
    v0_bounds = [0, 1]

    # Market params
    S0 = 1
    r = 0
    q = 0

    data_nn['lambda'] = np.random.uniform(lambd_bounds[0], lambd_bounds[1], n_samples)
    data_nn['vbar'] = np.random.uniform(vbar_bounds[0], vbar_bounds[1], n_samples)
    data_nn['eta'] = np.random.uniform(eta_bounds[0], eta_bounds[1], n_samples)
    data_nn['rho'] = np.random.uniform(rho_bounds[0], rho_bounds[1], n_samples)
    data_nn['v0'] = np.random.uniform(v0_bounds[0], v0_bounds[1], n_samples)

    data_nn['iv'] = data_nn.apply(lambda row: heston_pricer(row['lambda'], row['vbar'], row['eta'],
                                                            row['rho'], row['v0'], r, q,
                                                            row['Time to Maturity (years)'], S0, row['Moneyness'])[1],
                                  axis=1)

    # Drop ```NaN``` data

    data_nn.dropna(inplace=True)

    # Split generated labeled data into ```train```, ```val``` and ```test```

    # data_nn = data_nn.iloc[:990000, :]
    data_nn.reset_index(drop=True, inplace=True)
    data_train, data_val, data_test = np.split(data_nn, [int(9e5), int(9.45e5)], axis=0)

    # Store splitted data to local files
    data_train.to_csv("./data/heston/train.csv", index=False)
    data_val.to_csv("./data/heston/val.csv", index=False)
    data_test.to_csv("./data/heston/test.csv", index=False)

    # ## Generate rBergomi data

    # Define rBergomi pricer with Cholesky decomposition method, and Monte Carlo simulation

    # ### Generate rBergomi labeled data

    # Load previously generated $(m,T)$ data

    K_T = pd.read_csv("./data/strike_maturity.csv", index_col=0)
    n_samples = K_T.shape[0]

    # Market params
    S0 = 1.

    # Define rBergomi parameter generator with ```scipy.stats.truncnorm```
    # Generate labeled data
    # **NB**: The next cells are executable, but it takes too long a time to generate $10^6$ samples,
    # so in practice we generated only half of that on a cloud-based virtual machine.

    data_nn = K_T.merge(K_T.progress_apply(
        lambda row: pd.Series(
            generate_rBergomi_sample(row['Moneyness'], row['Time to Maturity (years)'], param_generator, S0)),
        axis=1), left_index=True, right_index=True)

    data_nn.dropna(inplace=True)
    data_nn.to_csv("./data/rBergomi/labeled_data_all.csv", index=False)
    # data_nn = data_nn.iloc[:990000, :]
    data_nn.reset_index(drop=True, inplace=True)
    data_train, data_val, data_test = np.split(data_nn, [int(9e5), int(9.5e5)], axis=0)
    data_train.to_csv("./data/heston/train.csv", index=False)
    data_val.to_csv("./data/heston/val.csv", index=False)
    data_test.to_csv("./data/heston/test.csv", index=False)
# # References


# [1] https://github.com/ryanmccrickerd/rough_bergomi
# 
# [2] https://github.com/amuguruza/RoughFCLT/blob/master/rDonsker.ipynb
