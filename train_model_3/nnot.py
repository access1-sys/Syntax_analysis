import csv
import numpy as np

with open("paper50_all2.csv",'r',encoding='ISO-8859-1') as f:
    reader=csv.reader(f)
    data=[row for row in reader]
data=np.array(data)

res=0
f=open('null_text','a',encoding='utf-8')
for row in data[:3500]:
    if row[2]=='null':
        f.write(row[1]+'\n')
        res+=1
        print(row[1])
f.close()
print(res)