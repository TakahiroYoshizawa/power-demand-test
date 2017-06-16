# -*-coding:utf-8-*-

import FFT_Forecast as ftf

len_forecast = int(input("どれだけの長さ予測しますか？ > "))
df = ftf.read_power_csv(csvname = './data/demand/juyo-2017.csv')
ts, log_mean = ftf.make_tswave(df)
X, fft_data  = ftf.do_fft(ts)
Lasso_coef = ftf.est_in_lasso(X, ts)
new = ftf.forcast(Lasso_coef, log_mean, fft_data, len_forecast)
print(df)
print(new)