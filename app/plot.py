import pandas as pd
import numpy as np

race_empty_series = pd.Series(data=[0.0, 0.0, 0.0, 0.0], index=['Other', 'Black', 'Hispanic', 'White'])
age_bins = [0, 25, 35, 45, 55, 65, 100]
age_bins_lab = ['..25', '26..35', '36..45', '46..55', '56..65', '66..']
year_bins = [0, 1985, 1990, 1995, 2000, 2005, 2010, 2015]
year_bins_lab = ['..85', '86..90', '91..95', '96..00', '01..05', '06..10', '11..15']


def bar_plot(df):
    race_series = df.groupby('race').size()
    race_series = race_series.add(race_empty_series, fill_value=0.0)
    race_series = (race_series / race_series.sum()) * 100.0
    race_series = np.round(race_series, decimals=2)

    lab1 = race_series.index.tolist()
    val1 = race_series.tolist()

    val2, lab2 = np.histogram(df.age, bins=age_bins)
    val2 = (val2 / float(val2.sum())) * 100.0
    val2 = np.round(val2, decimals=2)

    val3, lab3 = np.histogram(df.year, bins=year_bins)
    val3 = (val3 / float(val3.sum())) * 100.0
    val3 = np.round(val3, decimals=2)

    return {
        'race': {'lab': lab1, 'val': val1},
        'age': {'lab': age_bins_lab, 'val': val2.tolist()},
        'year': {'lab': year_bins_lab, 'val': val3.tolist()}
    }

def score_plot(df):
    return {
        'lab': [""] * len(df),
        'val': df.sim.tolist()
    }