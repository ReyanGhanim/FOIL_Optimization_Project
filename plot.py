import matplotlib as plt 
import numpy as np
import csv

x = []

# with open('Cropped_Data_Labels.csv', 'r', encoding="latin-1") as f:
#     # Create a CSV reader object
#     reader = csv.reader(f)

#     for row in reader:
#         # Print the row
#         x.append(row)

#         with open('Cropped_Data.csv', 'r', encoding="latin-1") as f:
#             # Create a CSV reader object
#             reader = csv.reader(f)

#             for row in reader:
#                 # Print the row
#                 x.append(row)

# print(x)



with open('Cropped_Data_Labels.csv', 'r', encoding="latin-1") as file1:
    with open('Cropped_Data.csv', 'r', encoding="latin-1") as file2:
        reader1 = csv.reader(file1)
        reader2 = csv.reader(file2)

        for row1, row2 in zip(reader1, reader2):
            x.append((row1, row2))

print(x)


