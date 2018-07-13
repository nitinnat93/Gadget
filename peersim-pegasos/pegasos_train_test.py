#Run Pegasos algorithm on each node's training set and save model files.

import os.path,subprocess
from subprocess import STDOUT,PIPE
import os
import shlex #Used to split commands correctly
import pandas as pd

peersim_path = '/home/nitin/Documents/Pegasos4/dsvm/peersim-pegasos/'
pegasos_native_path = '/home/nitin/Documents/Pegasos4/dsvm/jni-pegasos/src/pegasos-native'
reuters_exp_path = os.path.join(peersim_path,'data','reuters','reuters_experiments.csv')
reuters_exp_final_path = os.path.join(peersim_path,'data','reuters','reuters_experiments_final.csv')
pegasos_train_test_path = os.path.join(peersim_path,'data','reuters','pegasos_train_test.csv')
data_folder = os.path.join(peersim_path,'data/reuters')
train_prefix = 't_'
test_prefix = 'tst_'
max_iter = 500
lambda1 = 0.000136


df = pd.DataFrame(columns = ['lambda', 'max_iter', 'node_id', 
								'pegasos_norm','pegasos_avg_loss',
								'pegasos_zero_one','pegasos_primal_obj',
								'pegasos_loss_on_test','pegasos_zero_one_on_test'])
count = 0
for node_id in range(10):
	pegasos_train_path = os.path.join(data_folder,train_prefix + str(node_id) + '.dat')
	pegasos_test_path = os.path.join(data_folder,test_prefix + str(node_id) + '.dat')
	model_path = os.path.join(data_folder,'m_' + str(node_id))

	cmd = shlex.split('/home/nitin/Documents/Pegasos4/dsvm/jni-pegasos/src/pegasos-native/./pegasos -testFile '
		+ pegasos_test_path + ' -modelFile ' +  model_path
		+ ' -lambda ' + str(lambda1) + ' -iter ' + str(max_iter) 
		+ ' ' + pegasos_train_path)	
	
	
	output = subprocess.run(cmd,stdout=PIPE)
	#print(output)
	stuff = str(output.stdout).split('\n')
	
	stuff2 = stuff[0].split('\\n')
	#print(stuff2)
	for i,line in enumerate(stuff2):
		
		#print("Started iterating")
		if 'Time for training' in line:
			print("Entered if")
			pegasos_norm = float(stuff2[i+2].split(' = ')[0])
			pegasos_avg_loss = float(stuff2[i+3].split(' = ')[0])
			pegasos_zero_one = float(stuff2[i+4].split(' = ')[0])
			pegasos_primal_obj = float(stuff2[i+5].split(' = ')[0])
			pegasos_loss_on_test = float(stuff2[i+6].split(' = ')[0])
			pegasos_zero_one_on_test = float(stuff2[i+7].split(' = ')[0])
			df.loc[count] = [lambda1,max_iter, node_id, pegasos_norm,pegasos_avg_loss, 
								pegasos_zero_one, pegasos_primal_obj,pegasos_loss_on_test,
									pegasos_zero_one_on_test]
				
			count += 1


	
#reuters_exp_df = pd.read_csv(reuters_exp_path)
#final_df = pd.merge(reuters_exp_df, df, on=['lambda','max_iter','node_id'])
#final_df.set_index('node_id')
df.to_csv(pegasos_train_test_path)