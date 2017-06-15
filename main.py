# -*-coding:utf-8-*-

import FFT_Forecast as ftf

df = ftf.read_power_csv(csvname = './data/demand/juyo-2017.csv')
ts = ftf.make_tswave(df)
X  = ftf.do_fft(ts)
ftf.est_in_lasso(X, ts)
print(X)