import os
import pandas as pd
import numpy as np

PEERSIM_PATH = '../peersim-pegasos/'
dataset_folder_path = os.path.join(PEERSIM_PATH,'data/',"adult")

def plot_results(dataset_folder_path):
	#Now to plot results and save them.
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