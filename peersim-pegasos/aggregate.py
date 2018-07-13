#!/usr/bin/env python3

import os.path
import os
import pandas as pd
import argparse
from sys import exit
import numpy as np
#Aggregator script, loads from respective files,
#selects time, primal objective and error on test
#Load time from gadget_train.csv

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset name as found in ./data/",
	type = str, required = True)
args = parser.parse_args()
dataset = args.dataset

peersim_path = '../peersim-pegasos/'
pegasos_native_path = '../jni-pegasos/src/pegasos-native'

if not os.path.exists(peersim_path) or not os.path.exists(pegasos_native_path):
	print("The peersim and pegasos-native paths need to be changed as per your machine. Exiting...")
	exit()

#Setting file paths.

gadget_train_exp_file = 'gadget_train.csv'
dataset_folder_path = os.path.join(peersim_path,'data/',dataset)
gadget_test_exp_path = os.path.join(peersim_path,'data',dataset,'gadget_test_experiments.csv')
pegasos_test_exp_path = os.path.join(peersim_path,'data',dataset,'pegasos_test.csv')
gadget_train_path = os.path.join(dataset_folder_path,gadget_train_exp_file)

#Read time, lambda and iterations from gadget_train
agg_df = pd.DataFrame()
df = pd.read_csv(gadget_train_path)
agg_df["total_time"] = df["total_time"]
agg_df["max_iter"] = df["max_iter"]
agg_df["lambda"] = df["lambda"]
agg_df["node_id"] = df["node_id"]

#Read pegasos and gadget test experiments
df2 = pd.read_csv(gadget_test_exp_path)

df3 = pd.read_csv(pegasos_test_exp_path)

#Merge with original
agg_df = pd.merge(agg_df, df2, how = "left", on = ['node_id','lambda','max_iter'])
agg_df = pd.merge(agg_df, df3,how = "left" ,on = ['node_id', 'lambda', 'max_iter'])
agg_df["gadget_calc_obj"] = (agg_df["lambda"]/2)*agg_df["gadget_test_wt_norm"] * \
							agg_df["gadget_test_wt_norm"] + agg_df["gadget_test_loss"]
agg_df["pegasos_calc_obj"] = (agg_df["lambda"]/2)*agg_df["pegasos_test_wt_norm"] * \
							agg_df["pegasos_test_wt_norm"] + agg_df["pegasos_test_loss"]
#Need to group by node_id, lambda and max_iter
#Need to take averages of total_time, zero_one_test_error and calc_obj
agg_df2 = agg_df.groupby(['lambda','max_iter'])['lambda','max_iter','total_time','gadget_test_zero_one',
							'pegasos_test_zero_one','gadget_calc_obj','pegasos_calc_obj'].mean()
print(agg_df2.head())
agg_df.to_csv(os.path.join(dataset_folder_path,"all_results.csv"))
agg_df2.to_csv(os.path.join(dataset_folder_path,"aggregated_results.csv"))


def plot_results(dataset_folder_path):
	#Now to plot results and save them.
	import matplotlib
	matplotlib.use('agg')
	import matplotlib.pyplot as plt
	#Need to plot objective value over iterations
	aggregated_df = pd.read_csv(os.path.join(dataset_folder_path,"aggregated_results.csv"))
	plt.plot(np.array(aggregated_df["total_time"]/1000), 
		np.array(aggregated_df["pegasos_test_zero_one"]*100), 'r')
	plt.plot(np.array(aggregated_df["total_time"]/1000), 
		np.array(aggregated_df["gadget_test_zero_one"]*100), 'b')
	plt.legend(['Pegasos zero one error', 'Gadget zero one error'], loc='upper right')
	plt.xlabel("Average total time (in seconds)")
	plt.ylabel("Zero one test error (%)")
	plt.savefig(os.path.join(dataset_folder_path, "test_error_plot.png"))


	plt.figure()
	plt.plot(np.array(aggregated_df["total_time"]/1000), 
		np.array(aggregated_df["pegasos_calc_obj"]), 'r')
	plt.plot(np.array(aggregated_df["total_time"]/1000), 
		np.array(aggregated_df["gadget_calc_obj"]), 'b')
	plt.legend(['Pegasos objective', 'Gadget objective'], loc='upper right')
	plt.xlabel("Average total time (in seconds)")
	plt.ylabel("Objective value")

	plt.savefig(os.path.join(dataset_folder_path, "test_objective_plot.png"))

plot_results(dataset_folder_path)