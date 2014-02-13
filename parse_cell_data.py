import csv

with open('cells_example_1.csv', 'r') as csvfile:
    with open('cells.txt', 'w') as outfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == 0: continue
            key = str(i)
            val = ' '.join(row)
            outfile.write('%s %s\n' % (key, val))
        
