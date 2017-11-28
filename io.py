import csv
with open('data.csv') as f:
    reader = csv.reader(f, delimiter=';', quotechar='|')
    for row in reader:
        if row[0]=='26/10/2017':
            print(row[1])
