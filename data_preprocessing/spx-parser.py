# # SPX Raw Data Parser
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter
import sys
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import statsmodels.nonparametric.api as smnp
from IPython.display import display, clear_output


def plot_heatmap(data, values='Inv Spread'):
    """Plots the heatmap of the `values` column of a data frame 
    with respect to `Time to Maturity (years)` and `Log Moneyness`.
    """
    data_sort = data.sort_values(values, ascending=True)
    data_pivot = data_sort.pivot(index='Time to Maturity (years)', columns='Log Moneyness', values=values)
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(data_pivot, cmap=plt.cm.Spectral, cbar=True, xticklabels=data_pivot.columns.values.round(2),
                     yticklabels=data_pivot.index.values.round(2))
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # spot price is alongs wih the SPX prices data file, which is 2878.99 for the 9th April, 2019
    spot_price = 2878.99
    today = pd.to_datetime('20190409', format='%Y%m%d', errors='coerce')
    nb_trading_days = 365
    raw_spx = pd.read_csv("./data/SPX_20190409.csv", skiprows=2)
    raw_spx = raw_spx[raw_spx['Calls'].apply(lambda s: s.startswith('SPXW'))]
    raw_spx['Expiration Date'] = pd.to_datetime(raw_spx['Expiration Date'], format='%m/%d/%Y')
    raw_spx['Time to Maturity (years)'] = raw_spx['Expiration Date'].apply(
        lambda exp_date: (exp_date - today).days / nb_trading_days)
    raw_spx['Log Moneyness'] = np.log(raw_spx['Strike'] / spot_price)

    # ### Seperate ```calls``` and ```puts```, remove useless columns

    call_data = raw_spx.loc[:, ['Bid', 'Ask', 'IV', 'Open Int', 'Time to Maturity (years)', 'Log Moneyness', 'Strike']]
    put_data = raw_spx.loc[:,
               ['Bid.1', 'Ask.1', 'IV.1', 'Open Int.1', 'Time to Maturity (years)', 'Log Moneyness', 'Strike']]
    put_data.columns = call_data.columns

    call_data['Mid'] = (call_data['Bid'] + call_data['Ask']) / 2
    put_data['Mid'] = (put_data['Bid'] + put_data['Ask']) / 2
    # ### Store the processed calls and puts data to local file

    call_data.to_csv("./data/processed_spx_calls_all_liquids.csv")
    put_data.to_csv("./data/processed_spx_puts_all_liquids.csv")

    # - Difference between ```Ask``` and ```Bid```

    call_data = call_data[call_data['Open Int'] != 0]
    put_data = put_data[put_data['Open Int'] != 0]

    call_data['Inv Spread'] = 1. / (call_data['Ask'] - call_data['Bid'])
    put_data['Inv Spread'] = 1. / (put_data['Ask'] - put_data['Bid'])
    plot_heatmap(call_data, 'Inv Spread')
    plot_heatmap(put_data, 'Inv Spread')

    # **Observation**:
    # 
    # The liquidity is localized in a small region. According the proposition in the paper, we consider only the region:
    # 
    # - $-0.1 \leq \text{Log Moneyness} \leq 0.28$
    # 
    # - $\frac{1}{365} \leq \text{Time to Maturity} \leq 0.2$

    liquid_criterion = (-0.1 <= call_data['Log Moneyness']) & (call_data['Log Moneyness'] <= 0.28) & (
                1 / 365 <= call_data['Time to Maturity (years)']) & (call_data['Time to Maturity (years)'] <= 0.2)
    call_data_liquid = call_data[liquid_criterion].copy()
    plot_heatmap(call_data_liquid, 'Inv Spread')

    call_data_liquid.sort_values('Inv Spread', inplace=True, ascending=False)
    call_data_liquid.to_csv("./data/spx_liquid_calls.csv")
    call_data_liquid.head()

    mat_vals = call_data_liquid['Time to Maturity (years)'].unique()

    call_data_liquid[call_data_liquid['Time to Maturity (years)'] == mat_vals[0]].plot(x='Log Moneyness', y='Mid')


    # Load the previously processed call options data

    data = pd.read_csv("./data/processed_spx_calls_all_liquids.csv", index_col=0)

    # Count the ```Open Interests``` of each option

    total_interest = data['Open Int'].sum()
    interests = pd.DataFrame(index=np.arange(total_interest), columns=['Time to Maturity (years)', 'Log Moneyness'],
                             dtype=float)
    counter = 0
    for idx in data.index:
        num_int = data.loc[idx, 'Open Int']
        end_counter = counter + num_int
        interests.iloc[counter:end_counter, :] = data.loc[idx, ['Time to Maturity (years)', 'Log Moneyness']].values
        counter = end_counter

    fig, ax = plt.subplots()
    ax.set_title("Bivariate distribution estimation of log-moneyness and time-to-maturity")
    x = interests['Log Moneyness'].values
    y = interests['Time to Maturity (years)'].values
    ax = sns.kdeplot(x, y, cbar=True, shade=True, shade_lowest=False, cmap='BuPu')

    # User scikit-learn KDE to estimate the joint distribution of $\mathcal{K}_{(m,T)}$

    K_T = data[['Time to Maturity (years)', 'Log Moneyness']].values
    K_T[:, 1] = np.exp(K_T[:, 1])

    # grid search cv for best bandwidth
    params = {'bandwidth': np.logspace(-3, -1, 5)}
    grid = GridSearchCV(KernelDensity(), params, verbose=sys.maxsize, n_jobs=-1, cv=5)
    grid.fit(K_T)
    kde = grid.best_estimator_
    kde.fit(K_T)
    n_samples = int(1e6)
    generated_K_T = pd.DataFrame(index=np.arange(n_samples), columns=['Moneyness', 'Time to Maturity (years)'],
                                 dtype=float)
    counter = 0

    while counter < n_samples:
        clear_output()
        display("Generated {}% valid samples".format(counter / n_samples * 100))
        still_need = n_samples - counter
        samples = kde.sample(still_need)
        is_valid = (samples[:, 0] > 0.75) & (samples[:, 0] < 1.2) & (samples[:, 1] > 0) & (samples[:, 1] < 0.25)
        valid_samples = samples[is_valid]
        n_valid = len(valid_samples)
        new_counter = counter + n_valid
        generated_K_T.iloc[counter:new_counter, :] = valid_samples
        counter = new_counter

    # Store the generated $(m,T)$ pairs to local file for later use.
    generated_K_T.to_csv("./data/strike_maturity.csv")
