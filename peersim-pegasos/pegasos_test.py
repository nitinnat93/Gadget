#!/usr/bin/env python3

import os.path,subprocess
from subprocess import STDOUT,PIPE
import os
import shlex #Used to split commands correctly
import pandas as pd
import argparse
from sys import exit
#--------------------------Add command line arguments ------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset name as found in ./data/",
	type = str, required = True)
parser.add_argument("--reg_lambda", help="Regularization parameter lambda",type = float,
	nargs='?', const=1, default=0.000136)
parser.add_argument("--max_iter", help="Max number of iterations used for training the data.",
	type = int,nargs='?', const=1, default=500)
parser.add_argument("--override_file", help="If 1, the csv file is overwritten",
	type = int,nargs='?', const=1, default=0)
args = parser.parse_args()
lambda1 = args.reg_lambda
dataset = args.dataset
max_iter = args.max_iter
override = args.override_file




#--------------------------Setting paths ------------------------#

#The following two paths change based on machine.
peersim_path = '../peersim-pegasos/'
pegasos_native_path = '../jni-pegasos/src/pegasos-native'

if not os.path.exists(peersim_path) or not os.path.exists(pegasos_native_path):
	print("The peersim and pegasos-native paths need to be changed as per your machine. Exiting...")
	exit()


#File paths to store output csv files.
pegasos_test_exp_path = os.path.join(peersim_path,'data',dataset,'pegasos_test.csv')
pegasos_train_exp_path = os.path.join(peersim_path,'data',dataset,'pegasos_train.csv')
data_folder = os.path.join(peersim_path,'data/' + dataset)
train_prefix = 't_'
test_prefix = 'tst_'
model_prefix = 'm_'




#--------------------------Running local models on test set ------------------------#
df = pd.DataFrame(columns = ['max_iter','lambda','node_id', 'pegasos_test_wt_norm', 
								'pegasos_test_objective','pegasos_test_loss',
								'pegasos_test_zero_one'])
count = 0


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
			df.loc[count] = [max_iter,lambda1, node_id, wts_norm_on_test,obj_value_on_test, 
								loss_on_test,
								zero_one_on_test]
				
			count += 1
df.set_index('node_id')

#If override flag is 1 or the file path does not exist, overwrite.
if override == 1 or not os.path.exists(pegasos_test_exp_path):
	df.to_csv(pegasos_test_exp_path,index=False)
else:
	#Retrieve dataframe from the csv file, append to it and then write it back.
	df_temp = pd.read_csv(pegasos_test_exp_path)
	df_temp = df_temp.append(df,ignore_index=True)
	df_temp.to_csv(pegasos_test_exp_path,index=False)

print("Testing results of Pegasos on the test set can be found in " + pegasos_test_exp_path)

#--------------------------Running local models on training set ------------------------#

train_df = pd.DataFrame(columns = ['max_iter','lambda','node_id', 'pegasos_train_wt_norm', 
								'pegasos_train_objective','pegasos_train_loss',
								'pegasos_train_zero_one'])
count = 0

for node_id in range(10):
	

	testfile = os.path.join(data_folder,'t_' + str(node_id)+'.dat')
	modelfile = os.path.join(data_folder,'m_' + str(node_id) + '.dat')
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
			train_df.loc[count] = [max_iter,lambda1, node_id, wts_norm_on_train,obj_value_on_train, 
			loss_on_train,zero_one_on_train]
				
			count += 1
train_df.set_index('node_id')

#If override flag is 1 or the file path does not exist, overwrite.
if override == 1 or not os.path.exists(pegasos_train_exp_path):
	train_df.to_csv(pegasos_train_exp_path,index=False)
else:
	#Retrieve dataframe from the csv file, append to it and then write it back.
	df_temp = pd.read_csv(pegasos_train_exp_path)
	df_temp = df_temp.append(train_df,ignore_index=True)
	df_temp.to_csv(pegasos_train_exp_path,index=False)

print("Testing results of Pegasos on the train set can be found in " + pegasos_train_exp_path)
	