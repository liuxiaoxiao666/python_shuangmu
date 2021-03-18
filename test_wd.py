import pandas as pd
import os
import csv
import numpy as np
fnlist=os.listdir("wd_test")
f = open('twd.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
for fn in fnlist:
    df = pd.read_csv("wd_test/"+fn)
    print(fn)
    zzc=max(df['tag 0 z'])-min(df['tag 0 z'])
    data=[]
    data.append(df['tag 0 x'][0])
    data.append(df['tag 0 z'][0])
    data.append(zzc)
    data.append(np.var(df['tag 0 z'])*10000)
    print(data)
    csv_writer.writerow(data)
f.close()