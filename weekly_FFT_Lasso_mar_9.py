# -*-coding:utf-8-*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.fftpack as fft
from matplotlib import rc
from math import pi as pi
import sklearn.linear_model as sklm
import time
import sklearn.cross_validation as skcv
from collections import Counter



def read_data():
    """
    データを読み込む
    :return:data
    """
    data = pd.read_csv("~/PycharmProjects/MATSUO/daily_category_sales_quantity.csv", parse_dates=True, header=None)
    data.columns = ["date", "catg_cd", "catg", "sales", "qty"]
    return data

def make_data(data,code):
    """
    codeごとに学習データを作成plt.plot(fftYlist)
    plt.show()
    qty_dataに格納
    """
    df_by_code = data[data["catg_cd"] == code]
    df_by_code = df_by_code.set_index("date")
    df_by_code.index = pd.to_datetime(df_by_code.index)

    cteg_data = df_by_code[(df_by_code.index.year >= 2007) & (df_by_code.index.year <= 2013)]
    weekly_catg_data = cteg_data["qty"].resample('W', how='sum', closed='left', label='left')
    sum_sales = sum(cteg_data["sales"])
    sum_qty = sum(cteg_data["qty"])

    ave = sum_sales/sum_qty
    bst = [sum_qty,sum_sales,ave]
    qty_data = weekly_catg_data.drop(list(weekly_catg_data.index)[0])
    qty_data = qty_data.reset_index()
    qty_train = qty_data[:313]

    qty_test = qty_data["qty"][-52:-1]
    qty_test = qty_test.reset_index()

    return qty_data, qty_train, qty_test, bst


def forecast1():
    """
    この関数では、ログを取り、トレンド項を削除する。
    :param i:
    :return:
    """

    qty_train["logged_qty"] = np.log(qty_train["qty"])

    x = list(qty_train.index)
    x = sm.add_constant(x)
    y = qty_train["logged_qty"]
    model = sm.OLS(y, x)
    results = model.fit()
    intercept, x1 = results.params
    pred = qty_train.index * x1 + intercept
    Y = qty_train["logged_qty"] - pred
    lenY = len(Y)
    Lasso_X, fftY, freqs, power, phase, lenY = FFT(Y,lenY)

    #print("Mean", np.mean(np.abs(fftY)))
    #print("variance",np.var(fftY))


    qty_forecast,score = do_lasso(Y, Lasso_X, fftY, freqs, power, phase, lenY)
    qty_forecast = qty_forecast + intercept + [i * x1 for i in range(0, lenY+51)]
    qty_forecast = np.exp(qty_forecast)

    qty_forecast = qty_forecast[-51:]
    return qty_forecast, score

def FFT(Y,lenY):
    """
    フーリエ変換
    :param Y:
    :param lenY:
    :return:
    """
    fftY = fft.fft(Y)
    freqs = fft.fftfreq(len(Y))
    power = np.abs(fftY)
    phase = [np.arctan2(float(c.imag), float(c.real)) for c in fftY]

    Lasso_X = make_lassoX(fftY, freqs, power, phase, lenY)

    return Lasso_X, fftY, freqs, power, phase, lenY

def make_lassoX(fftY, freqs, power, phase, lenY):
    """
    周波数ごとに波を作成しLasso_Xを作成する
    :param fftY:
    :param freqs:
    :param power:
    :param phase:
    :param lenY:
    :return: Lasso_X
    """
    for po, fr, ph in zip(power, freqs, phase):
        average = []
        i = 0
        for t in range(0, lenY):
            average.append(po * np.cos((2 * pi * fr) * t + ph) / (lenY))
            i += 1
        Lasso_X[fr] = average

    return Lasso_X

def do_lasso(Y, Lasso_X, fftY, freqs, power, phase, lenY):

    print("__________________________________")

    lcv = sklm.LassoCV(eps=1e-2, n_alphas=100, cv=10, n_jobs=-1, fit_intercept=False)
    lcvf = lcv.fit(Lasso_X, Y)
    print("lcvf.alpha: ", lcvf.alpha_)

    Lasso_model = sklm.Lasso(alpha=0)
    results = Lasso_model.fit(Lasso_X, Y)

    #print(results.coef_)
    score = results.score(Lasso_X, Y)
    #print("score: ", score)
    coef = results.coef_
    data_loss = 0.5 * ((Lasso_X.dot(coef) - Y) ** 2).sum()
    n_samples, n_features = Lasso_X.shape
    penalty = n_samples * results.alpha * np.abs(coef).sum()
    likelihood = np.exp(-(data_loss + penalty))

    print("likelihood: ", likelihood)
    print("logL: ", np.log(likelihood))

    # result.conf_ の０じゃない要素数をカウント
    # result.conf_ の要素数
    # 要素ごとに条件にあうかどうかの T or Fを出すコード
    #print(np.array(results.coef_ != 0))
    hoge = np.array(results.coef_ != 0)

    #print(hoge)

    #print("datelength: " + str(len(results.coef_)))
    #print("count not0 is True or 0 is False: " + str(Counter(hoge)))
    print("__________________________________")

    plt.figure()
    plt.plot((results.coef_ * power),label="Using Lasso",lw=2.5)
    plt.plot(power,label="Without Lasso", lw=2.5)
    plt.legend()
    #plt.savefig("graph" + str(k) + "_6.pdf")

    plt.show()

    #powers = pd.DataFrame([list(results.coef_), list(power)])
    #powers.to_csv("powers"+str(k)+".csv")
    qty_forecast = []
    for t in range(0, lenY+51):
        average = 0
        for coef, po, fr, ph in zip(results.coef_, power, freqs, phase):
            average += coef * po * np.cos((2 * pi * fr) * t + ph) / (lenY+51)
        qty_forecast.append(average)

    return qty_forecast,score

def evaluate():
    #ウィークリー評価
    wmape = sum(abs(forecast-qty_test["qty"])/qty_test["qty"]) / 51 * 100

    reg = forecast - qty_test["qty"]
    preg = list(reg)
    mreg = list(reg)
    i=0
    for i in range(0,51):
        if reg[i]>=0:
            mreg[i] = 0
        elif reg[i]<0:
            preg[i] = 0
        i +=1
    preg = np.array(preg)
    mreg = np.array(mreg)
    wpmape = sum(abs(preg[0:51])/qty_test["qty"][0:51]) / 51 * 100
    wmmape = sum(abs(mreg[0:51])/qty_test["qty"][0:51]) / 51 * 100


    test = qty_test["qty"][0:51]
    #print("weekly: ", item)
    print("MAPE: ", wmape)
    #print("PlusMAPE", wpmape)
    #print("MinusMAPE", wmmape)

    evaluation = [wmape, score]

    return evaluation,test

if __name__ == '__main__':
    start = time.time()
    data = read_data()
    codeList = ["1111","1101", "1102", "1103", "1104", "1105",
                "1107", "1108", "1109", "1110", "1111",
                "1220", "1221", "1222", "1223", "1224",
                "1226", "1227", "1228", "5330", "5331", "5332"]
    submit = pd.DataFrame(None)
    evaluation_submit = pd.DataFrame(None)
    tests = pd.DataFrame(None)
    bstt = pd.DataFrame(None)

    k = 1
    for item in codeList:
        qty_data, qty_train, qty_test, bst = make_data(data, code=item)
        #qty_train.plot()
        #plt.show()
        forecast = []
        print(k)
        Lasso_X = pd.DataFrame(None)
        forecast, score = forecast1()
        submit[k] = forecast
        evaluation, test = evaluate()
        bstt[k] = bst
        evaluation_submit[k] = evaluation
        tests[k] = test
        k += 1
    elapsed_time = time.time() - start
    #print(elapsed_time)
    submit.columns = codeList
    evaluation_submit.columns = codeList
    bstt.columns = codeList
    tests.columns = codeList
    evaluationList = ["wmape","score"]
    evaluation_submit.index = evaluationList


    #submit.to_csv('forecast_cv610.csv')
    #evaluation_submit.to_csv('evaluation_cv610.csv')
    #tests.to_csv('test_3_19.csv')
    #bstt.to_csv('bstt.csv')
