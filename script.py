import csv
import numpy as np
 
 
with open('training_data.csv','r') as csvfile:
  data=list(csv.reader(csvfile))
  for i in range(1,14001):
      name=data[i][0]
      l=len(name)
      name=name[0:l-4]
      name=name+'.txt'
      s=str(data[i][5])+" "+str(data[i][6])+" "+str(data[i][7])+" "+str(data[i][8])+" "+str(data[i][9])
      f=open(name,'w+')
      f.write(s)
      f.close()