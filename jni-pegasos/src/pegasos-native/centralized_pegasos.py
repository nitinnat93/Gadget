#Run Pegasos algorithm on each node's training set and save model files.

import os.path,subprocess
from subprocess import PIPE
import os
import shlex #Used to split commands correctly
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse

data_folder = "/user/nitinnat/Pegasos4/dsvm/peersim-pegasos/data/"
exec_path = "./pegasos "
result_folder = "./results/"

dataset_dic = {'reuters': {'trainfile':'money-fx.trn',
                           'testfile':'money-fx.tst',
                           'lambda':0.000129,
                           'max_iter':100000
                           },
                'adult': {'trainfile':'adult.trn',
                          'testfile':'adult.tst',
                          'lambda':0.0000307,
                           'max_iter':100000
                          },
                'cov2': {'trainfile':'covtype_bn_train.trn',
                          'testfile':'covtype_bn_test.tst',
                          'lambda':0.000001,
                           'max_iter':100000
                          },
                'ccat': {'trainfile':'ccat.trn',
                          'testfile':'ccat.tst',
                          'lambda':0.0001,
                           'max_iter':100000
                          },
                'mnist': {'trainfile':'mnist.trn',
                          'testfile':'mnist.tst',
                          'lambda':0.0000167,
                           'max_iter':100000
                          },
                'usps': {'trainfile':'usps.trn',
                          'testfile':'usps.tst',
                          'lambda':0.000136,
                           'max_iter':100000
                          },
                'banana': {'trainfile':'banana.trn',
                          'testfile':'banana.tst',
                          'lambda':0.000001,
                           'max_iter':100000
                          },
                'waveform': {'trainfile':'waveform.trn',
                          'testfile':'waveform.tst',
                          'lambda':0.000001,
                           'max_iter':100000
                          }
                       
             }
        
#datasets = ['reuters','adult','cov2','ccat','mnist','usps'];
#trainfiles = ['money-fx.trn',
#         'adult.trn',
#         'covtype_bn_train.trn',
#         'ccat.trn',
#         'mnist.trn',
#         'usps.trn'
#         ]        
#testfiles = ['money-fx.tst',
#         'adult.tst',
#         'covtype_bn_test.tst',
#         'ccat.tst',
#         'mnist.tst',
#         'usps.tst'
#         ]     
#lambdas = [1.29 * 1e-4,
#           3.07 * 1e-5,
#           1e-6,
#           1e-4,
#           1.67 * 1e-5,
#           1.36 * 1e-4
#           ]
#
#max_iters = [2100,
#             2100,
#             2100,
#             2100,
#             2100,
#             2100
#             ]

#Train time per iter in iter 10000: 402 || Objective value in iter 10000: 0.18585 || Test loss in iter 10000: 0.120015 || Test error in iter 10000: 0.0327372

                             

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", help="Max iterations",type = int,required=True)
    args = parser.parse_args()
    #max_iters = [args.iters for i in range(len(datasets))]
    for key in dataset_dic.keys():
        dataset_dic[key]['max_iter'] = args.iters
    count = 0
    for d in sorted(['adult','ccat','cov2','mnist','reuters','usps']):
        result_file_path = os.path.join(result_folder,d + ".csv")
        train_path = os.path.join(data_folder, d,dataset_dic[d]['trainfile'])
        test_path = os.path.join(data_folder, d,dataset_dic[d]['testfile'])
        model_path = os.path.join(result_folder,d + "_model.dat")
        cmd = shlex.split(exec_path + '-testFile '
            + test_path + ' -modelFile ' +  model_path
            + ' -lambda ' + str(dataset_dic[d]['lambda']) + ' -iter ' + str(dataset_dic[d]['max_iter']) 
            + ' ' + train_path)    
        
        
        output = subprocess.run(cmd,stdout=PIPE)
        stuff = str(output.stdout).split('\n')
        stuff2 = stuff[0].split('\\n')
        ReadTime = 0
        df = pd.read_csv(os.path.join(result_folder,d + '_model.dat.csv'))
        for j,line in enumerate(stuff2):
            if 'Time for reading' in line:
                ReadTime = line.split(' = ')[0]                
        df['ReadTime'] = [ReadTime]*len(df)
        df.to_csv(os.path.join(result_folder,d + '_model.dat.csv'))
        """
        Plotting.
        Criteria: 1 legend only.
        Only the top row should have 
        
        
        plt.figure()
        plt.plot(df["TrainTime"].astype(float).tolist(),df["ObjValue"].astype(float).tolist())
        plt.savefig(os.path.join(result_folder,d+"_objvalue_plot.jpg"))
        plt.figure()
        plt.plot(df["TrainTime"].astype(float).tolist(),df["Epsilon"].astype(float).tolist())
        plt.savefig(os.path.join(result_folder,d+"_epsilon_plot.jpg"))
        plt.figure()
        plt.plot(df["TrainTime"].astype(float).tolist(),df["TestError"].astype(float).tolist())
        plt.savefig(os.path.join(result_folder,d+"_test_error_plot.jpg"))
        print(d + " done.")
        """



