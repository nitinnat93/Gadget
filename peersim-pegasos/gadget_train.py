#!/usr/bin/env python3
#Author: Nitin Nataraj, University at Buffalo

import os.path,subprocess
from subprocess import STDOUT,PIPE
import os
import shlex #Used to split commands correctly
import pandas as pd
import argparse


#--------------------------Add command line arguments ------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--reg_lambda", help="Set regularization parameter lambda.",type = float,
	 nargs='?', const=1, default=0.000136)
parser.add_argument("--max_iter", help="Set max number of iterations.",type = int,
	 nargs='?', const=1, default=500)
parser.add_argument("--override_file", help="If 1, the csv file is overwritten",
	type = int,nargs='?', const=1, default=0)
parser.add_argument("--dataset", help="Dataset name as found in ./data/",
	type = str,required =True)

args = parser.parse_args()
lambda1 = args.reg_lambda
max_iter = args.max_iter
override = args.override_file
dataset = args.dataset

peersim_path = '../peersim-pegasos/'
pegasos_native_path = '../jni-pegasos/src/pegasos-native'

dataset_folder_path = os.path.join(peersim_path,'data',dataset)
dataset_exp_file = 'gadget_train.csv'


if not os.path.exists(peersim_path) or not os.path.exists(pegasos_native_path):
	print("The peersim and pegasos-native paths need to be changed as per your machine. Exiting...")
	exit()

#--------------------------Creating dataframes to store info ------------------------#

df = pd.DataFrame(columns = ['lambda', 'max_iter', 'node_id', 
							 'norm_solution','gadget_train_loss',
							 'gadget_train_zero_one_error',
							 'gadget_train_objective','local_time'])
gadget_df = pd.DataFrame(columns = ['lambda', 'max_iter', 
									'node_id', 'gadget_wt_norm',
									'gadget_time'
									])
global_wt_df = pd.DataFrame(columns = ['lambda', 'max_iter', 
										'node_id', 'gadget_train_wt_norm'])


count,count2,count3 = 0,0,0

#Function to take a line and replace it with new content
def process(line):
		if "network.node.lambda" in line:
			return line.replace(line, "network.node.lambda " + str(lambda1) + '\n')
		elif "network.node.maxiter" in line:
			return line.replace(line, "network.node.maxiter " + str(max_iter) + '\n')
		elif "protocol.1.lambda" in line:
			return line.replace(line, "protocol.1.lambda " + str(lambda1) + '\n')
		elif "network.node.resourcepath" in line:
			return line.replace(line, "network.node.resourcepath " + dataset_folder_path + '\n')
		else:
			return line    	



#------------Replace configuration file with new params ------#

print("Replacing configuration file contents with new parameters...")
import fileinput
fileToSearch = "./config/config-pegasos2.cfg"

with fileinput.FileInput(fileToSearch, inplace=1) as file:
	for line in file:
		print(process(line),end='')


#--------------------------Begin training GADGET ------------------------#
print("Running command to train GADGET...")
cmd = shlex.split('java -cp "lib/*:classes" -Djava.library.path=lib peersim.Simulator config/config-pegasos2.cfg data/"outputdata.txt"')	
print("Reading data...")
output = subprocess.run(cmd,stdout=PIPE)	
stuff = str(output.stdout).split('\n')
stuff2 = stuff[0].split('\\n')

for i,line in enumerate(stuff2):
	if 'created node with ID' in stuff2[i]:	
		node_id = int(line.split(': ')[1])
		local_construction_time = float(stuff2[i-2].split()[-1])
		primal_obj = float(stuff2[i-4].split(' = ')[0])
		avg_zero_one_error = float(stuff2[i-5].split(' = ')[0])
		avg_loss_of_solution = float(stuff2[i-6].split(' = ')[0])
		norm_of_solution = float(stuff2[i-7].split(' = ')[0])
		df.loc[count] = [lambda1,max_iter, node_id, norm_of_solution,avg_loss_of_solution, 
							avg_zero_one_error, primal_obj,local_construction_time]
		count += 1
	if 'GADGET Norm' in stuff2[i]:
		node_id = int(stuff2[i].split(']')[-2].split('[')[-1])
		gadget_wt_norm = float(stuff2[i].split(':')[-1])
		misclassified = int(stuff2[i+1].split(' : ')[-1])
		gadget_time = float(stuff2[i-1].split()[-1])

		gadget_df.loc[count2] = [lambda1, max_iter,
							node_id, gadget_wt_norm,
							gadget_time
						]
		count2 += 1
	if '[finish]' in stuff2[i]:
		node_id = int(stuff2[i].split(']')[-2].split('[')[-1])
		global_weight_norm = float(stuff2[i].split()[-1])
		global_wt_df.loc[count3] = [lambda1, max_iter, 
									node_id, global_weight_norm]
		count3 += 1


#--------------------------Merge dataframes ------------------------#
final_df = pd.merge(df, gadget_df, on=['lambda','max_iter','node_id'])
final_df = pd.merge(final_df, global_wt_df, on=['lambda','max_iter','node_id'])
final_df["total_time"] = final_df["gadget_time"] + final_df["local_time"]
final_df["calc_obj"] = (lambda1/2)*final_df["gadget_wt_norm"]*final_df["gadget_wt_norm"] + final_df["gadget_train_loss"]
print(final_df.to_string(index = False))



#If override flag is 1 or the file path does not exist, overwrite.
if override == 1 or not os.path.exists(os.path.join(dataset_folder_path,dataset_exp_file)):
	final_df.to_csv(os.path.join(dataset_folder_path,dataset_exp_file),index=False)
else:
	#Retrieve dataframe from the csv file, append to it and then write it back.
	df_temp = pd.read_csv(os.path.join(dataset_folder_path,dataset_exp_file))
	df_temp = df_temp.append(final_df,ignore_index=True)
	df_temp.to_csv(os.path.join(dataset_folder_path,dataset_exp_file),index=False)


print("Processing complete. Training information is stored in " + 
	os.path.join(dataset_folder_path,dataset_exp_file))


