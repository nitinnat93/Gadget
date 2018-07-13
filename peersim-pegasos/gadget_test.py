#!/usr/bin/env python3
#Run Pegasos algorithm on each node's training set and save model files.

#Gadget algorithm run on test files using global models generated in the training phase
import os.path,subprocess
from subprocess import STDOUT,PIPE
import os
import shlex #Used to split commands correctly
import pandas as pd
import argparse
from sys import exit


#--------------------------Add command line arguments ------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--reg_lambda", help="Set regularization parameter lambda.",type = float,
	 nargs='?', const=1, default=0.000136)
parser.add_argument("--max_iter", help="Set max number of iterations.",type = int,
	 nargs='?', const=1, default=500)
parser.add_argument("--dataset", help="Dataset name as found in ./data/",
	type = str, required = True)
parser.add_argument("--override_file", help="If 1, the csv file is overwritten",
	type = int,nargs='?', const=1, default=0)

args = parser.parse_args()
lambda1 = args.reg_lambda
max_iter = args.max_iter
dataset = args.dataset
override = args.override_file
#--------------------------Setting paths ------------------------#
#The following two paths change based on machine. Change before running.

peersim_path = '../peersim-pegasos/'
pegasos_native_path = '../jni-pegasos/src/pegasos-native'

if not os.path.exists(peersim_path) or not os.path.exists(pegasos_native_path):
	print("The peersim and pegasos-native paths need to be changed as per your machine. Exiting...")
	exit()

#Setting file paths.
gadget_test_exp_path = os.path.join(peersim_path,'data',dataset,'gadget_test_experiments.csv')
gadget_train_exp_path = os.path.join(peersim_path,'data',dataset,'gadget_train_experiments.csv')
data_folder = os.path.join(peersim_path,'data/' + dataset)
train_prefix = 't_'
test_prefix = 'tst_'
model_prefix = 'global_'


#--------------------------Running global models on test set ------------------------#
df = pd.DataFrame(columns = ['max_iter','lambda','node_id', 'gadget_test_wt_norm', 
								'gadget_test_objective','gadget_test_loss',
								'gadget_test_zero_one'])
count = 0
print("Executing on the test set...")
for node_id in range(10):
	
	
	testfile = os.path.join(data_folder,test_prefix + str(node_id)+'.dat')
	modelfile = os.path.join(data_folder,model_prefix + str(node_id) + '.dat')
	cmd = shlex.split('bash testClassification.sh ' + testfile + ' ' + modelfile +' ' + str(lambda1))	
	
	output = subprocess.run(cmd,stdout=PIPE)
	print("Finished testing " + modelfile.split('/')[-1] + " on " + testfile.split('/')[-1])
	stuff = str(output.stdout).split('\n')
	stuff2 = stuff[0].split('\\n')

	for i,line in enumerate(stuff2):
		
		if 'Weights Norm' in line:
			wts_norm_on_test = float(line.split('\\t')[-1])
			obj_value_on_test = float(stuff2[i+1].split('\\t')[-1])
			loss_on_test = float(stuff2[i+2].split('\\t')[-1])
			zero_one_on_test = float(stuff2[i+3].split('\\t')[-1].split()[-1])
			df.loc[count] = [max_iter,lambda1, node_id, wts_norm_on_test,obj_value_on_test, loss_on_test,
								zero_one_on_test]
				
			count += 1
df.set_index('node_id')

#If override flag is 1 or the file path does not exist, overwrite.
if override == 1 or not os.path.exists(gadget_test_exp_path):
	df.to_csv(gadget_test_exp_path,index=False)
else:
	#Retrieve dataframe from the csv file, append to it and then write it back.
	df_temp = pd.read_csv(gadget_test_exp_path)
	df_temp = df_temp.append(df,ignore_index=True)
	df_temp.to_csv(gadget_test_exp_path,index=False)



#--------------------------Running global models on training set ------------------------#
gadget_run_on_train = pd.DataFrame(columns = ['max_iter','lambda','node_id', 'gadget_train_wt_norm', 
								'gadget_train_objective','gadget_train_loss',
								'gadget_train_zero_one'])
count = 0
print("Executing on the train set...")
for node_id in range(10):
	

	testfile = os.path.join(data_folder,train_prefix + str(node_id)+'.dat')
	modelfile = os.path.join(data_folder,'global_' + str(node_id) + '.dat')
	cmd = shlex.split('bash testClassification.sh ' + testfile + ' ' + modelfile +' ' + str(lambda1))	
	
	output = subprocess.run(cmd,stdout=PIPE)
	print("Finished testing " + modelfile.split('/')[-1] + " on " + testfile.split('/')[-1])
	stuff = str(output.stdout).split('\n')
	stuff2 = stuff[0].split('\\n')

	for i,line in enumerate(stuff2):
		
		if 'Weights Norm' in line:
			wts_norm_on_train = float(line.split('\\t')[-1])
			obj_value_on_train = float(stuff2[i+1].split('\\t')[-1])
			loss_on_train = float(stuff2[i+2].split('\\t')[-1])
			zero_one_on_train = float(stuff2[i+3].split('\\t')[-1].split()[-1])
			gadget_run_on_train.loc[count] = [max_iter,lambda1, node_id, wts_norm_on_train,
												obj_value_on_train, loss_on_train,zero_one_on_train]
				
			count += 1
gadget_run_on_train.set_index('node_id')

#If override flag is 1 or the file path does not exist, overwrite.
if override == 1 or not os.path.exists(gadget_train_exp_path):
	gadget_run_on_train.to_csv(gadget_train_exp_path,index=False)
else:
	#Retrieve dataframe from the csv file, append to it and then write it back.
	df_temp = pd.read_csv(gadget_train_exp_path)
	df_temp = df_temp.append(gadget_run_on_train,ignore_index=True)
	df_temp.to_csv(gadget_train_exp_path,index = False)


	