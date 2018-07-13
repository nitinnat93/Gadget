#Run on data and split it into n files
import numpy as np
train_prefix = 't_'
test_prefix = 'tst_'
import os

#peersim_path = '/home/nitin/Documents/Pegasos4/dsvm/peersim-pegasos/'
#reuters_folder_path = os.path.join(peersim_path,'data','reuters')
#pegasos_native_path = '/home/nitin/Documents/Pegasos4/dsvm/jni-pegasos/src/pegasos-native'
#reuters_train_path = os.path.join(reuters_folder_path,'money-fx.trn')
#reuters_test_path = os.path.join(reuters_folder_path,'money-fx.tst') 
n = 10

def normalize(arr):
    return list(np.array((arr - np.min(arr))/(np.max(arr) - np.min(arr))))
#Read file
def write_to_file(filepath,writepath,prefix,n):
    with open(filepath,'r') as f:
        contents = f.read()
        datapoints = contents.split('\n')
        label = []
        for i in range(len(datapoints)):
            dp = datapoints[i].split()
            label.append(dp[0])
            
        print(datapoints[0])
        
        #label = [datapoints[i].split()[0] for i in range(len(datapoints))]
        sparse_input = [datapoints[i].split()[1:] for i in range(len(datapoints))]
        
        sparse_input1 = []
        sparse_input2 = []
        
        for i,lis in enumerate(sparse_input):
            temp1 = [item.split(':')[0] + ':' for item in lis]
            temp2 = normalize([float(item.split(':')[1]) for item in lis])
            sparse_input1.append(temp1)
            sparse_input2.append(temp2)
            datapoints[i] = ' '.join([temp1[j] + str(temp2[j]) for j in range(len(temp2))])

        datapoints = [label[i] +' '+ datapoints[i] for i in range(len(datapoints))] 
        print(datapoints[0])
        #sparse_input = ' '.join([sparse_input_1[i] + str(sparse_input_2[i]) for i in range(len(sparse_input))])
        #print(sparse_input[0])
        
        pts_per_file = int(round(len(datapoints)/n))
    for i in range(n):
        batch = datapoints[i*pts_per_file:min(len(datapoints)-1,(i+1)*pts_per_file)]
        #Write to file
        with open(os.path.join(writepath,prefix + str(i) + '.dat'), 'w') as wfile:
            wfile.write('\n'.join(batch))
    print("Data into " + str(n) +" files.")

train_path = "./covtype_bn_train.trn"
test_path = "./covtype_bn_test.tst"
write_to_file(train_path,'./' ,train_prefix,n)
write_to_file(test_path, './',test_prefix,n)

