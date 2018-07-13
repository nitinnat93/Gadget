# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 18:29:10 2018

@author: Nitin
"""

#!/usr/bin/env python3
#Author: Nitin Nataraj, University at Buffalo

import os.path,subprocess
from subprocess import PIPE
import os
import shlex #Used to split commands correctly
import pandas as pd
import argparse
import fileinput #To replace contents of a file inplace
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

class Experiment:
    def __init__(self,
                 dataset,
                 run,
                 peersim_path = '../peersim-pegasos/',
                 pegasos_native_path = '../jni-pegasos/src/pegasos-native',
                 override = 0,
                 config_file_path = './config/config-pegasos2.cfg',
                 train_prefix = 't_',
                 test_prefix = 'tst_',
                 local_model_prefix = 'm_',
                 global_model_prefix = 'global_'
                 ):
        if not os.path.exists(peersim_path) or not os.path.exists(pegasos_native_path):
            print("The peersim and pegasos-native paths don't exist. Exiting...")
            exit()
        self.peersim_path = peersim_path
        self.pegasos_native_path = pegasos_native_path
        self.dataset = dataset
        self.dataset_folder_path = os.path.join(self.peersim_path,'data',self.dataset)
        self.override = override
        self.config_file_path = config_file_path
        self.num_files = 10
        self.train_exp_path = os.path.join(peersim_path,'data',dataset,dataset + '_train_experiments.csv')
        self.gadget_train_exp_path = os.path.join(peersim_path,'data',dataset,dataset + '_gadget_train_experiments.csv')
        self.gadget_test_exp_path = os.path.join(peersim_path,'data',dataset,dataset + '_gadget_test_experiments.csv')
        self.pegasos_test_exp_path = os.path.join(peersim_path,'data',dataset,dataset + '_pegasos_test_experiments.csv')
        self.pegasos_train_exp_path = os.path.join(peersim_path,'data',dataset,dataset + '_pegasos_train_experiments.csv')
        self.run = run #Which iteration is begin run?
        
        #Initializing dataframes to hold results
        self.train_df = None
        self.gadget_train_df = None
        self.gadget_test_df = None
        self.pegasos_test_df = None
        self.pegasos_train_df = None
        self.aggregated_df = None
        #Prefixes
        self.train_prefix = train_prefix
        self.local_model_prefix = local_model_prefix
        self.test_prefix = test_prefix
        self.global_model_prefix = global_model_prefix
        
#--------------------------Creating dataframes to store info ------------------------#
        
    def _create_train_dataframes(self):
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
        
        return df, gadget_df, global_wt_df
    #Function to take a line and replace it with new content
    def _process_line(self,line):
    		if "network.node.lambda" in line:
    			return line.replace(line, "network.node.lambda " + str(self.lambda1) + '\n')
    		elif "network.node.maxiter" in line:
    			return line.replace(line, "network.node.maxiter " + str(self.max_iter) + '\n')
    		elif "protocol.1.lambda" in line:
    			return line.replace(line, "protocol.1.lambda " + str(self.lambda1) + '\n')
    		elif "network.node.resourcepath" in line:
    			return line.replace(line, "network.node.resourcepath " + self.dataset_folder_path + '\n')
    		else:
    			return line
    
    def _modify_config(self):
        #------------Replace configuration file with new params ------#

        print("Replacing configuration file contents with new parameters...")
        with fileinput.FileInput(self.config_file_path, inplace=1) as file:
            for line in file:
                print(self._process_line(line),end='')
    def train_gadget(self,lambda1,max_iter):
        #--------------------------Begin training GADGET ------------------------#
        #Store these as class variables to be used in testing later
        self.lambda1 = lambda1
        self.max_iter = max_iter
        df, gadget_df, global_wt_df = self._create_train_dataframes()
        
        #Modify the config file
        self._modify_config()
        
        print("Running command to train GADGET...")
        cmd = shlex.split('java -cp "lib/*:classes" -Djava.library.path=lib peersim.Simulator ' + self.config_file_path +' data/"outputdata.txt"')	
        print("Reading data...")
        output = subprocess.run(cmd,stdout=PIPE)	
        stuff = str(output.stdout).split('\n')
        stuff2 = stuff[0].split('\\n')
        count,count2,count3 = 0,0,0
        
        for i,line in enumerate(stuff2):
            if 'created node with ID' in stuff2[i]:
                node_id = int(line.split(': ')[1])
                local_construction_time = float(stuff2[i-2].split()[-1])
                primal_obj = float(stuff2[i-4].split(' = ')[0])
                avg_zero_one_error = float(stuff2[i-5].split(' = ')[0])
                avg_loss_of_solution = float(stuff2[i-6].split(' = ')[0])
                norm_of_solution = float(stuff2[i-7].split(' = ')[0])
                df.loc[count] = [lambda1,max_iter, node_id, norm_of_solution,
                           avg_loss_of_solution,avg_zero_one_error, primal_obj,
                           local_construction_time
                           ]
                count += 1
            if 'GADGET Norm' in stuff2[i]:
                node_id = int(stuff2[i].split(']')[-2].split('[')[-1])
                gadget_wt_norm = float(stuff2[i].split(':')[-1])
                gadget_time = float(stuff2[i-1].split()[-1])
                gadget_df.loc[count2] = [lambda1, max_iter,node_id, 
                                  gadget_wt_norm,gadget_time
                                  ]
                count2 += 1
            if '[finish]' in stuff2[i]:
                node_id = int(stuff2[i].split(']')[-2].split('[')[-1])
                global_weight_norm = float(stuff2[i].split()[-1])
                global_wt_df.loc[count3] = [lambda1, max_iter, 
                                     node_id, global_weight_norm
                                     ]
                count3 += 1
            #--------------------------Merge dataframes ------------------------#
            final_df = pd.merge(df, gadget_df, on=['lambda','max_iter','node_id'])
            final_df = pd.merge(final_df, global_wt_df, on=['lambda','max_iter','node_id'])
            final_df["total_time"] = final_df["gadget_time"] + final_df["local_time"]
            final_df["calc_obj"] = (lambda1/2)*final_df["gadget_wt_norm"]*final_df["gadget_wt_norm"] + final_df["gadget_train_loss"]
            final_df["run"] = self.run
            self.train_df = final_df
            final_df.set_index('node_id')
            #Write or append to file
            #If override flag is 1 or the file path does not exist, overwrite.
            if self.override == 1 or not os.path.exists(self.train_exp_path):
                self.train_df.to_csv(self.train_exp_path,index=False)
            else:
                #Retrieve dataframe from the csv file, append to it and then write it back.
                df_temp = pd.read_csv(self.train_exp_path)
                df_temp = df_temp.append(final_df,ignore_index=True)
                df_temp.to_csv(self.train_exp_path,index=False)
                self.train_df = df_temp
        print("Processing complete. Training information is stored in " + self.train_exp_path)
        print(final_df.to_string(index = False))
            
    def _create_test_dataframes(self):
        #--------------------------Running global models on test set ------------------------#
        df = pd.DataFrame(columns = ['max_iter','lambda','node_id', 'gadget_test_wt_norm',
                                     'gadget_test_objective','gadget_test_loss',
                                     'gadget_test_zero_one'])
        gadget_run_on_train = pd.DataFrame(columns = ['max_iter','lambda','node_id', 'gadget_train_wt_norm',
                                                      'gadget_train_objective','gadget_train_loss',
                                                      'gadget_train_zero_one'])
        return df, gadget_run_on_train
    
    def test_gadget(self):
        count = 0
        df,gadget_run_on_train = self._create_test_dataframes()
        print("Executing on the test set...")
        for node_id in range(self.num_files):
            testfile = os.path.join(self.dataset_folder_path,self.test_prefix + str(node_id)+'.dat')
            modelfile = os.path.join(self.dataset_folder_path,self.global_model_prefix + str(node_id) + '.dat')
            cmd = shlex.split('bash testClassification.sh ' + testfile + ' ' + modelfile +' ' + str(self.lambda1))
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
                    df.loc[count] = [self.max_iter,self.lambda1, 
                          node_id, wts_norm_on_test,obj_value_on_test, loss_on_test,
                          zero_one_on_test]
                    count += 1
        df["run"] = self.run
        #If override flag is 1 or the file path does not exist, overwrite.
        if self.override == 1 or not os.path.exists(self.gadget_test_exp_path):
            df.to_csv(self.gadget_test_exp_path,index=False)
            self.gadget_test_df = df
        else:
            #Retrieve dataframe from the csv file, append to it and then write it back.
            df_temp = pd.read_csv(self.gadget_test_exp_path)
            df_temp = df_temp.append(df,ignore_index=True)
            df_temp.to_csv(self.gadget_test_exp_path,index=False)
            self.gadget_test_df = df_temp
        
        #Run gadget on training set
        count = 0
        print("Executing on the train set...")
        for node_id in range(self.num_files):
            testfile = os.path.join(self.dataset_folder_path,self.train_prefix + str(node_id)+'.dat')
            modelfile = os.path.join(self.dataset_folder_path,self.global_model_prefix + str(node_id) + '.dat')
            cmd = shlex.split('bash testClassification.sh ' + testfile + ' ' + modelfile +' ' + str(self.lambda1))	
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
                    gadget_run_on_train.loc[count] = [self.max_iter,self.lambda1, node_id, wts_norm_on_train,
                                           obj_value_on_train, loss_on_train,zero_one_on_train]
            				
                    count += 1
        #Append the iteration column
        gadget_run_on_train["run"] = self.run
        gadget_run_on_train.set_index('node_id')  
        #If override flag is 1 or the file path does not exist, overwrite.
        if self.override == 1 or not os.path.exists(self.gadget_train_exp_path):
            gadget_run_on_train.to_csv(self.gadget_train_exp_path,index=False)
            self.gadget_train_df = gadget_run_on_train
        else:
            #Retrieve dataframe from the csv file, append to it and then write it back.
            df_temp = pd.read_csv(self.gadget_train_exp_path)
            df_temp = df_temp.append(gadget_run_on_train,ignore_index=True)
            df_temp.to_csv(self.gadget_train_exp_path,index = False)
            self.gadget_train_df = df_temp
                
    def _create_pegasos_dataframes(self):
        #--------------------------Running local models on test set ------------------------#
        pegasos_test_df = pd.DataFrame(columns = ['max_iter','lambda','node_id', 'pegasos_test_wt_norm',
                                     'pegasos_test_objective','pegasos_test_loss',
                                     'pegasos_test_zero_one'])
        pegasos_train_df = pd.DataFrame(columns = ['max_iter','lambda','node_id', 'pegasos_train_wt_norm',
                                                   'pegasos_train_objective','pegasos_train_loss',
                                                   'pegasos_train_zero_one'])
        
        return pegasos_test_df, pegasos_train_df
    
    def test_pegasos(self):
        count = 0
        pegasos_test_df, pegasos_train_df = self._create_pegasos_dataframes()
        for node_id in range(self.num_files):        
            testfile = os.path.join(self.dataset_folder_path,self.test_prefix + str(node_id)+'.dat')
            modelfile = os.path.join(self.dataset_folder_path,self.local_model_prefix + str(node_id) + '.dat')
            cmd = shlex.split('bash testClassification.sh ' + testfile + ' ' + modelfile +' ' + str(self.lambda1))	
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
                        pegasos_test_df.loc[count] = [self.max_iter,
                                           self.lambda1, node_id, wts_norm_on_test,obj_value_on_test,
                                           loss_on_test,zero_one_on_test
                                           ]
                count += 1
        pegasos_test_df["run"] = self.run
        pegasos_test_df.set_index('node_id')
        #If override flag is 1 or the file path does not exist, overwrite.
        if self.override == 1 or not os.path.exists(self.pegasos_test_exp_path):
            pegasos_test_df.to_csv(self.pegasos_test_exp_path,index=False)
            self.pegasos_test_df = pegasos_test_df
        else:
            #Retrieve dataframe from the csv file, append to it and then write it back.
            df_temp = pd.read_csv(self.pegasos_test_exp_path)
            df_temp = df_temp.append(pegasos_test_df,ignore_index=True)
            df_temp.to_csv(self.pegasos_test_exp_path,index=False)
            self.pegasos_test_df = df_temp
        print("Testing results of Pegasos on the test set can be found in " + self.pegasos_test_exp_path)
        
        
        
        #Run local models on Pegasos trainset
        count = 0
        for node_id in range(self.num_files):
            testfile = os.path.join(self.dataset_folder_path,self.train_prefix + str(node_id)+'.dat')
            modelfile = os.path.join(self.dataset_folder_path,self.local_model_prefix + str(node_id) + '.dat')
            cmd = shlex.split('bash testClassification.sh ' + testfile + ' ' + modelfile +' ' + str(self.lambda1))	
        	
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
                    pegasos_train_df.loc[count] = [self.max_iter,self.lambda1, node_id, 
                                        wts_norm_on_train,obj_value_on_train, 
                                        loss_on_train,zero_one_on_train]
                    count += 1
        pegasos_train_df["run"] = self.run
        pegasos_train_df.set_index('node_id')
        #If override flag is 1 or the file path does not exist, overwrite.
        if self.override == 1 or not os.path.exists(self.pegasos_train_exp_path):
            pegasos_train_df.to_csv(self.pegasos_train_exp_path,index=False)
            self.pegasos_train_df = pegasos_train_df
        else:
            #Retrieve dataframe from the csv file, append to it and then write it back.
            df_temp = pd.read_csv(self.pegasos_train_exp_path)
            df_temp = df_temp.append(pegasos_train_df,ignore_index=True)
            df_temp.to_csv(self.pegasos_train_exp_path,index=False)
            self.pegasos_train_df = df_temp
        print("Testing results of Pegasos on the train set can be found in " + self.pegasos_train_exp_path)
    
    def aggregate(self):
        #Read time, lambda and iterations from gadget_train
        agg_df = pd.DataFrame()
        df = pd.read_csv(self.train_exp_path)
        agg_df["total_time"] = df["total_time"]
        agg_df["max_iter"] = df["max_iter"]
        agg_df["lambda"] = df["lambda"]
        agg_df["node_id"] = df["node_id"]
        agg_df["run"] = df["run"]
        
        #Read pegasos and gadget test experiments
        df2 = pd.read_csv(self.gadget_test_exp_path)
        
        df3 = pd.read_csv(self.pegasos_test_exp_path)
        
        #Merge with original
        agg_df = pd.merge(agg_df, df2, how = "left", on = ['node_id','lambda','max_iter','run'])
        agg_df = pd.merge(agg_df, df3,how = "left" ,on = ['node_id', 'lambda', 'max_iter','run'])
        agg_df["gadget_calc_obj"] = (agg_df["lambda"]/2)*agg_df["gadget_test_wt_norm"] * \
        							agg_df["gadget_test_wt_norm"] + agg_df["gadget_test_loss"]
        agg_df["pegasos_calc_obj"] = (agg_df["lambda"]/2)*agg_df["pegasos_test_wt_norm"] * \
        							agg_df["pegasos_test_wt_norm"] + agg_df["pegasos_test_loss"]
        #Need to group by node_id, lambda and max_iter
        #Need to take averages of total_time, zero_one_test_error and calc_obj
        agg_df2 = agg_df.groupby(['lambda','max_iter'])['total_time','gadget_test_zero_one','pegasos_test_zero_one',
                                'gadget_calc_obj','pegasos_calc_obj'].mean()
        #agg_df2 = agg_df2.groupby(['max_iter','lambda'])['lambda',
        #                        'total_time','gadget_test_zero_one','pegasos_test_zero_one',
        #                        'gadget_calc_obj','pegasos_calc_obj'].mean()
        print(agg_df2.head())
        agg_df.to_csv(os.path.join(self.dataset_folder_path,self.dataset + "_all_results.csv"))
        agg_df2.to_csv(os.path.join(self.dataset_folder_path,self.dataset +  "_aggregated_results.csv"))
        self.aggregated_df = agg_df2
    def plot_results(self):
        #Now to plot results and save them.
        #Need to plot objective value over iterations
        aggregated_df = pd.read_csv(os.path.join(self.dataset_folder_path,self.dataset+ "_aggregated_results.csv"))
        plt.plot(np.array(aggregated_df["total_time"]/1000), 
                 np.array(aggregated_df["pegasos_test_zero_one"]*100), 'r')
        plt.plot(np.array(aggregated_df["total_time"]/1000), 
                 np.array(aggregated_df["gadget_test_zero_one"]*100), 'b')
        plt.legend(['Pegasos zero one error', 'Gadget zero one error'], loc='upper right')
        plt.xlabel("Average total time (in seconds)")
        plt.ylabel("Zero one test error (%)")
        plt.savefig(os.path.join(self.dataset_folder_path,self.dataset +  "_test_error_plot.png"))
        plt.figure()
        plt.plot(np.array(aggregated_df["total_time"]/1000), 
                 np.array(aggregated_df["pegasos_calc_obj"]), 'r')
        plt.plot(np.array(aggregated_df["total_time"]/1000), 
                 np.array(aggregated_df["gadget_calc_obj"]), 'b')
        plt.legend(['Pegasos objective', 'Gadget objective'], loc='upper right')
        plt.xlabel("Average total time (in seconds)")
        plt.ylabel("Objective value")
        
        plt.savefig(os.path.join(self.dataset_folder_path, self.dataset + "_test_objective_plot.png"))
        print('Plots are saved in ' + self.dataset_folder_path)
    def _remove_useless_files(self):
        #Helper function used to removed unnecessary clutter created by the Java code
        ##--------------------Remove unnecessary files from dataset folder--------------##

        all_files = list(os.walk(self.dataset_folder_path))[0][2]
        files_to_remove = [file for file in all_files if "Ms_Cl" in file or 
                           'time_Vec' in file or 'Wt_Nm' in file]
        for f in files_to_remove:
            os.remove(os.path.join(self.dataset_folder_path,f))
        print("Unnecessary files have been removed.")



if __name__ == "__main__":
    #--------------------------Add command line arguments ------------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument("--reg_lambda", help="Set regularization parameter lambda.",type = float,
    	 nargs='?', const=1, default=0.000136)
    parser.add_argument("--dataset", help="Dataset name as found in ./data/",
    	type = str,required =True)
    parser.add_argument("--runs", help="Number of times the experiment needs to be run.",
    	type = int,nargs='?', const=1, default=10)
    
    args = parser.parse_args()
    lambda1 = args.reg_lambda
    dataset = args.dataset
    runs = args.runs
    #datasets = ['reuters', 'cov2','adult','ccat']
    override = 0
    #lambdas = [0.0000129, 0.000001, 0.0000307]#,0.0001]
    max_iters = [500*i + 1 for i in range(100)]
    ####----Driver code for the experiments----####

    for p in range(runs):
        for q, max_iter in enumerate(max_iters):
            print("We are in run %d and max_iter %d: " %(p,max_iter))
            if q == 0:
                #This is the first ever run of the first ever value of max_iter
                #Hence override should be set to 1
                override = 1
                experiment = Experiment(dataset = dataset, run = p, override = override)
                experiment.train_gadget(lambda1, max_iter)
                experiment.test_gadget()
                experiment.test_pegasos()
                override = 0 #We need to append from here on
            else:
                
                experiment = Experiment(dataset = dataset, run = p, override = override)
                experiment.train_gadget(lambda1, max_iter)
                experiment.test_gadget()
                experiment.test_pegasos()
                    
    experiment._remove_useless_files()
    experiment.aggregate()
    experiment.plot_results()
            
