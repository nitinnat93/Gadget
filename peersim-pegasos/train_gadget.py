# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 00:38:14 2018

@author: Nitin
"""
import os.path,subprocess
from subprocess import PIPE
import os
import shlex #Used to split commands correctly
import pandas as pd
import argparse
import fileinput
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

class Gadget:
    def __init__(self,
                 dataset,
                 run,
                 peersim_path = '/projects/academic/haimonti/Pegasos4/dsvm/peersim-pegasos/',
                 pegasos_native_path = '../jni-pegasos/src/pegasos-native',
                 config_file_path = '/projects/academic/haimonti/Pegasos4/dsvm/peersim-pegasos/config/config-pegasos2.cfg',
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
            elif "network.node.examperiter" in line:
                return line.replace(line, "network.node.examperiter " + '1' + '\n') #hardcoded for now
            else:
                return line
    def _remove_useless_files(self):
        #Helper function used to removed unnecessary clutter created by the Java code
        ##--------------------Remove unnecessary files from dataset folder--------------##

        all_files = list(os.walk(self.dataset_folder_path))[0][2]
        files_to_remove = [file for file in all_files if "Ms_Cl" in file or 
                           'time_Vec' in file or 'Wt_Nm' in file]
        for f in files_to_remove:
            os.remove(os.path.join(self.dataset_folder_path,f))
        print("Unnecessary files have been removed.")
    def _modify_config(self):
        #------------Replace configuration file with new params ------#

        print("Replacing configuration file contents with new parameters...")
        with fileinput.FileInput(self.config_file_path, inplace=1) as file:
            for line in file:
                print(self._process_line(line),end='')
    def train_gadget(self,lambda1,max_iter,override = 0):
            #--------------------------Begin training GADGET ------------------------#
            #Store these as class variables to be used in testing later
            self.lambda1 = lambda1
            self.max_iter = max_iter
            #df, gadget_df, global_wt_df = self._create_train_dataframes()
            
            #Modify the config file
            self._modify_config()
            
            print("Running command to train GADGET...")
            cmd = shlex.split('java -cp "lib/*:classes" -Djava.library.path=lib peersim.Simulator ' + self.config_file_path +' data/"outputdata.txt"')    
            print("Reading data...")
            output = subprocess.run(cmd,stdout=PIPE)
            stuff = str(output.stdout).split('\n')
            stuff2 = stuff[0].split('\\n')

            readingTimes = [float(line.split('=')[0]) for line in stuff2 if 'Reading time' in line]
            trainTimes = [float(line.split('=')[0]) for line in stuff2 if 'Model training time' in line]
            calcObjTimes = [float(line.split('=')[0]) for line in stuff2 if 'Time to calculate the objective' in line]
            avgLosses = [float(line.split('=')[0]) for line in stuff2 if 'avg Loss of solution' in line]
            primalObjectives = [float(line.split('=')[0]) for line in stuff2 if 'primal objective of solution' in line]
            zeroOneErrors = [float(line.split('=')[0]) for line in stuff2 if 'avg zero-one error' in line]
            convergenceEpsilons = [float(line.split('=')[0]) for line in stuff2 if 'Epsilon at convergence' in line] 
            convergenceIters = [float(line.split('=')[0]) for line in stuff2 if 'Convergence iteration' in line] 
            solutionNorms = [float(line.split('=')[0]) for line in stuff2 if 'Norm of solution' in line] 
            gadgetTimes = [float(line.split(' ')[-1]) for line in stuff2 if 'Time for running GADGET is' in line] 
            gadgetNorms =  [float(line.split(':')[-1]) for line in stuff2 if 'GADGET Norm' in line] 
            globalWtNorms = [float(line.split(':')[-1]) for line in stuff2 if 'global weight norm at node' in line] 
            numMisclassified = [float(line.split(':')[-1]) for line in stuff2 if '#misclassified at node' in line] 
            assert self.num_files == len(numMisclassified) == len(globalWtNorms) == len(convergenceIters)  
            assert self.num_files == len(gadgetTimes) == len(gadgetNorms) == len(readingTimes)  
            assert self.num_files == len(trainTimes) == len(calcObjTimes) == len(avgLosses) 
            assert self.num_files == len(primalObjectives) == len(zeroOneErrors) == len(convergenceEpsilons) == len(solutionNorms)
            
            ##Testing on the test set
            print("Executing on the test set...")
            all_test_output = []
            for node_id in range(self.num_files):
                testfile = os.path.join(self.dataset_folder_path,self.test_prefix + str(node_id)+'.dat')
                modelfile = os.path.join(self.dataset_folder_path,self.global_model_prefix + str(node_id) + '.dat')
                cmd = shlex.split('bash testClassification.sh ' + testfile + ' ' + modelfile +' ' + str(self.lambda1))
                output = subprocess.run(cmd,stdout=PIPE)
                stuff = str(output.stdout).split('\n')
                stuff2 = stuff[0].split('\\n')
                all_test_output += stuff2
            stuff2 = all_test_output #cringey assignment due to laziness
            testWtNorms = [float(line.split('\\t')[-1]) for line in stuff2 if 'Weights Norm Value' in line]
            testObjVals = [float(line.split('\\t')[-1]) for line in stuff2 if 'Objective Value' in line]
            testLosses = [float(line.split('\\t')[-1]) for line in stuff2 if 'Loss Value' in line]
            testZeroOneErrors = [float(line.split(' ')[-1]) for line in stuff2 if 'Zero One Error' in line]
            assert self.num_files == len(testWtNorms) == len(testObjVals) == len(testLosses) == len(testZeroOneErrors)
            
            df = pd.DataFrame({"Node" : list(range(0,self.num_files)),
                   "Max Iter" : [self.max_iter]*self.num_files,
                   "Lambda" : [self.lambda1]*self.num_files, 
                   "Reading Time": readingTimes,
                   "Training Time": trainTimes,
                   "Obj Calc Time": calcObjTimes,
                   "Average Loss": avgLosses,
                   "Primal Objective": primalObjectives,
                   "Zero One Error": zeroOneErrors,
                   "Convergence Epsilon": convergenceEpsilons,
                   "Convergence Iter": convergenceIters,
                   "Norm of Solution": solutionNorms,
                   "Gadget Norm": gadgetNorms,
                   "Gadget Time": gadgetTimes,
                   "Global Wt Norm":globalWtNorms,
                   "Num Misclassified": numMisclassified,
                   "Test Wt Norm": testWtNorms,
                   "Test Objective": testObjVals,
                   "Test Loss": testLosses,
                   "Test Zero One Error": testZeroOneErrors
                   })
            df["Total Time"] = df["Reading Time"] + df["Training Time"] + df["Obj Calc Time"]
            agg_df = df.groupby(["Max Iter"])["Max Iter","Lambda","Test Zero One Error","Primal Objective","Total Time"].mean()
            
            
            if override == 1 or not os.path.exists(self.train_exp_path):
                df.to_csv(self.train_exp_path,index=False)
                self.train_exp_path = df
                agg_df.to_csv(os.path.join(self.dataset_folder_path,self.dataset+"_final_results.csv"),index = False)
            else:
                #Retrieve dataframe from the csv file, append to it and then write it back.
                df_temp = pd.read_csv(self.train_exp_path)
                df_temp = df_temp.append(df,ignore_index=True)
                df_temp.to_csv(self.train_exp_path,index=False)
                self.train_df = df_temp
                
                df_temp = pd.read_csv(os.path.join(self.dataset_folder_path,self.dataset+"_final_results.csv"))
                df_temp = df_temp.append(agg_df,ignore_index=True)
                df_temp.to_csv(os.path.join(self.dataset_folder_path,self.dataset+"_final_results.csv"),index = False)
    def train_gadget_new(self,lambda1,max_iter,override = 0):
        #--------------------------Begin training GADGET ------------------------#
        #Store these as class variables to be used in testing later
        self.lambda1 = lambda1
        self.max_iter = max_iter
        #df, gadget_df, global_wt_df = self._create_train_dataframes()
        
        #Modify the config file
        self._modify_config()
        
        print("Running command to train GADGET...")
        cmd = shlex.split('java -cp "lib/*:classes" -Djava.library.path=lib peersim.Simulator ' + self.config_file_path +' data/"outputdata.txt"')    
        print("Reading data...")
        output = subprocess.run(cmd,stdout=PIPE)
        stuff = str(output.stdout).split('\n')
        stuff2 = stuff[0].split('\\n')
        gadgetTimes = [float(line.split()[-1]) for line in stuff2 if 'time for running gadget' in line.lower()]
        lossUpdateTimes = [float(line.split()[-1]) for line in stuff2 if 'time for loss updates' in line.lower()]
        #Read from the respective datafolder
        dataframes = []
        lengths = []
        plt.figure()
        for i in range(0,self.num_files):
            df_temp = pd.read_csv(os.path.join(self.dataset_folder_path,'m_' + str(i) + '.dat.csv'))
            dataframes.append(df_temp)
            lengths.append(len(df_temp))
            #Plot each one
            plt.plot(df_temp["TrainTime"], df_temp["TestError"])
        
            
        plt.savefig(os.path.join(self.dataset_folder_path,"test_error_plot.png"))
        min_length = min(lengths)
 
        #Make the following into a function later.
        #Clip all the dataframes to this size
        dataframes = [dataframes[i].iloc[0:min_length] for i in range(0,self.num_files)]
        assert len(dataframes[0]) == len(dataframes[self.num_files-1]) #assert the same lengths
        ##Now to calculate the averages across all nodes
        final_df = pd.DataFrame()
        for i in range(0,self.num_files):
            if i == 0:    
                final_df["ObjValue"] = dataframes[0]['ObjValue'].iloc[0:min_length]
                final_df["Epsilon"] = dataframes[0]["Epsilon"].iloc[0:min_length]
                final_df["CalcObjTime"] = dataframes[0]["CalcObjTime"].iloc[0:min_length]
                final_df["TrainTime"] = dataframes[0]["TrainTime"].iloc[0:min_length]
                final_df["TestLoss"] = dataframes[0]["TestLoss"].iloc[0:min_length]
                final_df["TestError"] = dataframes[0]["TestError"].iloc[0:min_length]
            else:
                final_df["ObjValue"] += dataframes[i]["ObjValue"].iloc[0:min_length]
                final_df["Epsilon"] += dataframes[i]["Epsilon"].iloc[0:min_length]
                final_df["CalcObjTime"] += dataframes[i]["CalcObjTime"].iloc[0:min_length]
                final_df["TrainTime"] += dataframes[i]["TrainTime"].iloc[0:min_length]
                final_df["TestLoss"] += dataframes[i]["TestLoss"].iloc[0:min_length]
                final_df["TestError"] += dataframes[i]["TestError"].iloc[0:min_length]
        final_df /= self.num_files
        final_df["GadgetTime"] = [np.mean(gadgetTimes)]*min_length
        final_df["Lambda"] = [self.lambda1]*min_length
        final_df["Iter"] = df_temp["Iter"].iloc[0:min_length] #Will be the same for all 10 nodes due to clipping.
        #Get reading times
        readingTimes = []
        for i in range(0,self.num_files):
            f = open(os.path.join(self.dataset_folder_path,'m_'+str(i) + '.dat.txt')).read()
            readingTimes.append(float(f.split(',')[0].split(' = ')[0]))
        assert len(readingTimes) == self.num_files
        
        final_df["ReadingTime"] = [np.mean(readingTimes)]*min_length
        #Total time taken is the time to calculate objective, reading time and the GADGET time
        final_df["TotalTime"] = final_df["ReadingTime"] + final_df["GadgetTime"] + final_df["CalcObjTime"] + final_df["TrainTime"]
        final_df.to_csv(os.path.join(self.dataset_folder_path,self.dataset+ '_agg_results.csv'),index = False)








if __name__ == "__main__":

    #--------------------------Add command line arguments ------------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument("--reg_lambda", help="Set regularization parameter lambda.",type = float,
         nargs='?', const=1, default=0.000136)
    parser.add_argument("--dataset", help="Dataset name as found in ./data/",
        type = str,required =True)
    parser.add_argument("--max_iter", help="Maximum iterations.",
        type = str,required =True)
    parser.add_argument("--configfile", help="Config file name as found in ./config/",
        type = str,required =True)
    """
    parser.add_argument("--runs", help="Number of times the experiment needs to be run.",
        type = int,nargs='?', const=1, default=10)
    
    parser.add_argument("--iters_per_round", help="Iters per round",
        type = int,required =True)
    
    args = parser.parse_args()
    lambda1 = args.reg_lambda
    dataset = args.dataset
    #runs = args.runs
    
    #iters_per_round = args.iters_per_round
    iters = args.iters
    #datasets = ['reuters', 'cov2','adult','ccat']
    override = 0
    #lambdas = [0.0000129, 0.000001, 0.0000307]#,0.0001]
    #max_iters = [iters_per_round*i + 100 for i in range(iters)]
    ####----Driver code for the experiments----####
#    
    for p in range(runs):
        for q, max_iter in enumerate(max_iters):

            print("We are in run %d and max_iter %d: " %(p,max_iter))
            if q == 0 and p == 0:
                #This is the first ever run of the first ever value of max_iter
                #Hence override should be set to 1
                override = 1
                experiment = Gadget(dataset = dataset, run = p, config_file_path = "./config/" + configfile)
                experiment.train_gadget(lambda1, max_iter,override)
                override = 0 #We need to append from here on
            else:
                
                experiment = Gadget(dataset = dataset, run = p, config_file_path = "./config/" + configfile)
                experiment.train_gadget(lambda1, max_iter,override)
    """
    args = parser.parse_args()
    lambda1 = args.reg_lambda
    dataset = args.dataset
    #runs = args.runs
    configfile = args.configfile
    #iters_per_round = args.iters_per_round
    iters = args.max_iter
    
    experiment = Gadget(dataset = dataset, run = 0, config_file_path = "/projects/academic/haimonti/Pegasos4/dsvm/peersim-pegasos/config/" + configfile)
    experiment.train_gadget_new(lambda1,iters,override=1)
    experiment._remove_useless_files()
