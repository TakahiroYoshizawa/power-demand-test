# おまじない
# -*-coding:utf-8-*-

# import packages
import pandas as pd
import numpy as np
from sklearn import linear_model as sklm
import scipy.fftpack as fft
from matplotlib import pyplot as plt
from math import pi
import time




# データ読み込み関数
def read_power_csv(csvname):
    '''
    :FUNCTION: 電力データの読み込み
    :return df
    '''
    df = pd.read_csv(csvname, skiprows=3, parse_dates=True, header=None)
    df.columns = ['date','time','power']
    return df

# FFTに投げる前の時系列データの取り出し
def make_tswave(df):
    '''
    :FUNCTION:log & delete trend
    :return t<s
    '''
    #logを取る
    log_ts = np.log(df['power'])[0:100]

    #indexを用いて回帰分析をする際に形を変えなければならない
    log_ts_index = np.array(log_ts.index).reshape(len(log_ts), 1)

    #indexと線形回帰をしてトレンド成分を推定
    trend_est_model = sklm.LinearRegression(fit_intercept = True)
    trend_est_model.fit(log_ts_index, log_ts)

    #残差を新たなtsとする
    ts = log_ts - trend_est_model.predict(log_ts_index)

    log_mean = trend_est_model.intercept_

    return ts, log_mean

# 時系列データに高速フーリエ変換をかけ，周波数成分へ落とし込む．
def do_fft(ts):
    '''
    :FUNCTION: fftをかけて，各周波数成分に分解する
    :param ts: 
    :return: X
    '''

    #フーリエ変換
    Y = fft.fft(ts)
    #周波数のリスト
    freqs = fft.fftfreq(len(ts))
    #位相の格納
    phase = [np.arctan2(float(c.imag), float(c.real)) for c in Y]
    #寄与率の算出
    power = np.abs(Y)

    #周波数データをデータフレームに格納
    fft_data = pd.DataFrame([power, freqs, phase]).T
    fft_data.columns = ['power', 'freqs', 'phase']

    #各行の周波数成分を用いて周波数ごとの波形を作成する
    X = fft_data.apply(lambda x: make_lassoX(x, len(fft_data)), axis=1).T
    return X, fft_data

def make_lassoX(x, T):
    '''
    :FUNCTION: 周波数ごとに，周波数成分を用いて波を生成
    :param x: 
    :param T: 
    :return: 
    '''

    Wave_by_freq = pd.Series([x['power'] * np.cos((2 * pi * x['freqs']) * t + x['phase']) / (len(x)) for t in range(T)])
    return Wave_by_freq

def est_in_lasso(X, ts):
    '''
    :FUNCTION: 
    :param X: 
    :param ts: 
    :return: 
    '''

    lassoCV = sklm.LassoCV(eps=1e-2, n_alphas=100, cv=10, n_jobs=-1, fit_intercept=False)
    lassoCV = lassoCV.fit(X, ts)

    Lasso_model = sklm.Lasso(alpha = lassoCV.alpha_)
    Lasso_model.fit(X, ts)

    score = Lasso_model.score(X, ts)
    coef = Lasso_model.coef_
    data_loss = 0.5 * ((X.dot(coef) - ts) ** 2).sum()
    n_samples, n_features = X.shape
    penalty = n_samples * Lasso_model.alpha * np.abs(coef).sum()
    likelihood = np.exp(-(data_loss + penalty))

    print("lassoCV.alpha: ", lassoCV.alpha_)
    print("score: ", score)
    print("likelihood: ", likelihood)
    print("logL: ", np.log(likelihood))


    return coef


def forcast(coef, intercept, fft_data, len_forcast):
    fft_data["coef"] = coef
    print(fft_data)
    L = len(fft_data) + len_forcast


    new = [sum(fft_data.apply(lambda x: make_forecast_by_freq(x,t), axis=1)) for t in range(L)]

    forecast = np.exp(new + intercept)

    return forecast

def make_forecast_by_freq(x, t):
    y = x['coef'] * x['power'] * np.cos((2 * pi * x['freqs']) * t + x['phase'])
    return y