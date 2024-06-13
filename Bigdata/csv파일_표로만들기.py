import matplotlib.pyplot as plt
import csv
labels = ['', '', '', '', '', '', '', '', '', '']
values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

csv_file = 'singer2.csv'

with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)

    cnt = 0
    for row in reader:
        labels[cnt] = row[0]
        values[cnt] = row[6]
        values[cnt] = int(row[6].replace(',', ''))**(1/2)
        cnt += 1

for i in range(10):
    for j in range(9-i):
        if values[j] < values[j+1]:
            temp = values[j]
            temp1 = labels[j]

            values[j] = values[j+1]
            labels[j] = labels[j+1]

            values[j+1] = temp
            labels[j+1] = temp1


plt.bar(labels, values, color='blue')

plt.show()

