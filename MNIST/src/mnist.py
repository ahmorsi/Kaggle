import csv as csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier

csv_file_object = csv.reader(open('./data/train.csv', 'rb')) 	# Load in the csv file
header = csv_file_object.next() 						# Skip the fist line as it is a header
data=[] 												# Create a variable to hold the data
labels = []
for row in csv_file_object: 							# Skip through each row in the csv file,
    labels.append(row[0])
    data.append(row[1:]) 								# adding each row to the data variable

data = np.array(data) 									# Then convert from a list to an array.
labels = np.array(labels)

print("Train")
forest = RandomForestClassifier(n_estimators=25)
forest = forest.fit(data, labels)

csv_file_object = csv.reader(open('./data/test.csv', 'rb')) 	# Load in the csv file
header = csv_file_object.next() 						# Skip the fist line as it is a header
test_data=[] 												# Create a variable to hold the data


for row in csv_file_object:
    test_data.append(row[0:])
test_data = np.array(test_data);

print("Predicating")
output = forest.predict(test_data).astype(int)

predictions_file = open("./submission/rndForest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ImageId", "Label"])
for idx,label in enumerate(output):
    open_file_object.writerow([idx+1,label])
predictions_file.close()
print('Done.')



