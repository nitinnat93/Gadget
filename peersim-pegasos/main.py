#Main file
import os
import shlex #Used to split commands correctly
import subprocess
from subprocess import PIPE
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--override_file", help="set to 1 to overwrite the output files",type = str,
	nargs='?', const=1, default="1")
parser.add_argument("--dataset", help="Dataset name as found in ./data/",
	type = str, required = True)
parser.add_argument("--reg_lambda", help="Set regularization parameter lambda.",
	type = float,required = True)
args = parser.parse_args()
override = args.override_file
dataset = args.dataset
lambda1= args.reg_lambda
peersim_path = '../peersim-pegasos/'
pegasos_native_path = '../jni-pegasos/src/pegasos-native'


gadget_train_path = os.path.join(peersim_path,"gadget_train.py")
gadget_test_path = os.path.join(peersim_path,"gadget_test.py")
pegasos_test_path = os.path.join(peersim_path,"pegasos_test.py")


#max_iters =[5, 25, 50, 75, 100, 125, 150, 200, 500, 750, 1000, 2000]
count = 0
max_iters = 1
time_limit = 10000
#Check for what all max_iter and lambda values, the experiments have been completed
max_iters_exist = None
try:
	df = pd.read_csv(os.path.join(peersim_path,'data',dataset,'aggregated_results.csv'))
	max_iters_exist = df["max_iter"]
except FileNotFoundError:
	pass


if override == '1':
	i=0
	cmd = shlex.split('python3 gadget_train.py --reg_lambda ' + str(lambda1) + 
		' --max_iter ' + str(max_iters) + " --override_file "+ override + " --dataset "+ dataset)
	output = subprocess.run(cmd,stdout=PIPE)	
	print("Ran gadget_train with max_iter = " + str(max_iters))

	#Now run gadget test

	cmd = shlex.split('python3 gadget_test.py --reg_lambda ' + str(lambda1) + 
		' --max_iter ' + str(max_iters) + " --override_file " + override + " --dataset "+ dataset)
	output = subprocess.run(cmd,stdout=PIPE)	
	print("Ran gadget_test with max_iter = " + str(max_iters))


	#Now run pegasos test

	cmd = shlex.split('python3 pegasos_test.py --reg_lambda ' + str(lambda1) + 
		' --max_iter ' + str(max_iters) + " --override_file " + override + " --dataset "+ dataset)
	output = subprocess.run(cmd,stdout=PIPE)	
	print("Ran pegasos_test with max_iter = " + str(max_iters))
#Checks time after each run and stops the loop if time goes above a limit
stop = False
override = '0'

#Keeps a track of the past five time values, and increases the max_iters drastically
#if there isn't much of an increase in time values.
prev_times = []
time_save_count = 0 #Reset after every 3 rounds  
while not stop:
	
	try:
		if len(prev_times) == 3:
			if prev_times[2] - prev_times[0] < 200:
				print("Stepping up max_iters.")
				max_iters += 300
				prev_times = []
				time_save_count = 0
		else:
			max_iters += 200
			time_save_count += 1
		cmd = shlex.split('python3 gadget_train.py --reg_lambda ' + str(lambda1) + 
			' --max_iter ' + str(max_iters) + " --override_file "+ override + " --dataset "+ dataset)
		output = subprocess.run(cmd,stdout=PIPE)	
		print("Ran gadget_train with max_iter = " + str(max_iters))
		#Check if the time has gone above time_limit
		times = pd.read_csv(os.path.join(peersim_path,'data',dataset,"gadget_train.csv"))["total_time"]
		current_max_time = times.iloc[-1]
		
		#Update the rolling time tracker
		prev_times.append(current_max_time)
		

		print("Max iters: %d" %(max_iters))
		print("Time reached: " + str(current_max_time))
		if current_max_time > time_limit:
			stop = True



		#Now run gadget test

		cmd = shlex.split('python3 gadget_test.py --reg_lambda ' + str(lambda1) + 
			' --max_iter ' + str(max_iters) + " --override_file " + override + " --dataset "+ dataset)
		output = subprocess.run(cmd,stdout=PIPE)	
		print("Ran gadget_test with max_iter = " + str(max_iters))


		#Now run pegasos test

		cmd = shlex.split('python3 pegasos_test.py --reg_lambda ' + str(lambda1) + 
			' --max_iter ' + str(max_iters) + " --override_file " + override + " --dataset "+ dataset)
		output = subprocess.run(cmd,stdout=PIPE)	
		print("Ran pegasos_test with max_iter = " + str(max_iters))
		print("Iterations updated to " + str(max_iters))
	except KeyboardInterrupt:
		stop = True

##--------------------Remove unnecessary files from dataset folder--------------##

all_files = list(os.walk(os.path.join(peersim_path,'data',dataset)))[0][2]
files_to_remove = [file for file in all_files if "Ms_Cl" in file or 
					'time_Vec' in file or
					'Wt_Nm' in file]
for f in files_to_remove:
	os.remove(os.path.join(peersim_path,'data',dataset,f))
print("Unnecessary files have been removed.")

##--------------------Aggregate results--------------##


cmd = shlex.split('python3 aggregate.py --dataset ' + dataset )
output = subprocess.run(cmd,stdout=PIPE)	
print("Aggregated the results.")