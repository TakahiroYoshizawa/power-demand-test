# -*-coding:utf-8-*-

import FFT_Forecast as ftf

df = ftf.read_power_csv(csvname = './data/demand/juyo-2017.csv')
ts = ftf.make_tswave(df)
X, fft_data  = ftf.do_fft(ts)
Lasso_coef = ftf.est_in_lasso(X, ts)

