import csv
import math

with open('results/measurement_it.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

smean1, smean2, sstd1, sstd2, svaria1, svaria2, sw1 = 0,0,0,0,0,0,0
counter_0 = 0

for i in range(0, len(data)):
	if data[i][1].find("it_4") != -1:
		counter_0 += 1
		smean2 += float(data[i][3])
		sstd2 += float(data[i][5])
		svaria2 += float(data[i][7])
		sw1 += float(data[i][8])

print(str(math.log(round(smean2/counter_0,8))))
print(str(math.log(round(sstd2/counter_0,8))))
print(str(math.log(round(svaria2/counter_0,8))))
print(str(math.log(round(sw1/counter_0,8))))
