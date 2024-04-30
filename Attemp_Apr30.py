import csv
import numpy as np

x = []
y = np.asarray(x)

with open('Cropped_Data.csv', 'r', encoding="latin-1") as f:
            
            # Create a CSV reader object
            reader = csv.reader(f)

            for row in reader:
                # Print the row
                y.np.append(row)


fitResults = fitMESI_TR(T, y**2)