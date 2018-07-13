#Run on data and split it into n files


import os
import argparse
import sys

#These prefixes will be used to name the split train and test files.
train_prefix = 't_'
test_prefix = 'tst_'

#Input dataset argument. NO default is provided. User has to input this argument.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset name as found in ./data/",
	type = str,required =True)
args = parser.parse_args()
dataset = args.dataset

peersim_path = '../peersim-pegasos/'
pegasos_native_path = '../jni-pegasos/src/pegasos-native'


dataset_folder_path = os.path.join(peersim_path,'data',dataset)

#Find the train and test files automatically. THEY HAVE TO BE LABELED .trn and .tst

all_data_files = list(os.walk(dataset_folder_path))[0][2]
data_train_path = [f for f in all_data_files if '.trn' in f]
data_test_path = [f for f in all_data_files if '.tst' in f]

if (not len(data_train_path) == 1) or (not len(data_test_path) == 1):
	print("There must be one file with extension .trn and one with .tst")
	print("Exiting...")
	sys.exit()

data_train_path = os.path.join(dataset_folder_path,data_train_path[0])
data_test_path = os.path.join(dataset_folder_path, data_test_path[0])

n = 10  #Parse this value from the config file later.

#Read file
def write_to_file(filepath,writepath,prefix,n):
    with open(filepath,'r') as f:
        contents = f.read()
        datapoints = contents.split('\n')
        if len(datapoints[-1].split()) == 0:
            print("Omitting last line")
            datapoints = datapoints[:-1]
        print(datapoints[-1])
        print("Total number of datapoints: %d" %(len(datapoints)))
        pts_per_file = int(round((len(datapoints)-1)/n))
        print("Total number of datapoints per file: %d" %(pts_per_file))
    for i in range(n):
        batch = datapoints[i*pts_per_file:min(len(datapoints)-1,(i+1)*pts_per_file)]
        #Write to file
        with open(os.path.join(writepath,prefix + str(i) + '.dat'), 'w') as wfile:
            wfile.write('\n'.join(batch))
    print("Split the data into " + str(n) +" files.")

write_to_file(data_train_path, dataset_folder_path,train_prefix,n)
write_to_file(data_test_path, dataset_folder_path,test_prefix,n)

