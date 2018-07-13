#Python Version 3
#Author: Nitin Nataraj, University at Buffalo

import os.path,subprocess
from subprocess import STDOUT,PIPE
import os
import shlex #Used to split commands correctly
import pandas as pd






cmd = shlex.split('java -cp "lib/*:classes" -Djava.library.path=lib peersim.Simulator config/config-pegasos2.cfg data/"outputdata.txt"')	
output = subprocess.run(cmd,stdout=PIPE)	

df = pd.DataFrame(columns = ['lambda', 'max_iter', 'node_id', 
								'norm_solution','avg_loss',
								'zero_one_error','primal_obj'])

stuff = str(output.stdout).split('\n')
stuff2 = stuff[0].split('\\n')
count = 0

for lambda1 in [0.01,0.03]:
	for max_iter in [1000,1002]:
		#We need to alter the config file here. We will modify the config-pegasos2.cfg file
		conf = open('./config/config-pegasos.cfg','r')
		conf_contents = conf.read()
		#Now replace the respective values with the parameters given in the iterators
		conf_contents = conf_contents.replace('network.node.lambda 0.01','network.node.lambda ' + str(lambda1))
		conf_contents = conf_contents.replace('network.node.maxiter 1000','network.node.maxiter ' + str(max_iter))
		print(conf_contents)
		conf.close()
		conf = open('./config/config-pegasos2.cfg','w')
		conf.write(conf_contents)
		conf.close()

		for i,line in enumerate(stuff2):
			if 'created node with ID' in stuff2[i] and int(line.split(': ')[1]) < 9:	
				node_id = int(line.split(': ')[1])
				norm_of_solution = float(stuff2[i+2].split(' = ')[0])
				avg_loss_of_solution = float(stuff2[i+3].split(' = ')[0])
				avg_zero_one_error = float(stuff2[i+4].split(' = ')[0])
				primal_obj = float(stuff2[i+5].split(' = ')[0])
				df.loc[count] = [lambda1,max_iter, node_id, norm_of_solution,avg_loss_of_solution, 
									avg_zero_one_error, primal_obj]
				count += 1
print(df.to_string(index = False))
df.to_csv('./data/rcv1_experiments.csv')



