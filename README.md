# Gadget


Requirements for standalone Pegasos:
1. C++ compiler - gcc-7.3.0
2. GNU Make 3.82

Requirements for Peersim:
1. gcc-7.3.0
2. ant-1.8.1
3. Java:
	openjdk version "1.8.0_171"
	OpenJDK Runtime Environment (build 1.8.0_171-b10)
	OpenJDK 64-Bit Server VM (build 25.171-b10, mixed mode)


To run the standalone pegasos code:
1. cd ./pegasos_svm
2. make
3. ./pegasos -testFile <test_file> -modelFile <model_file> -round <round_number> -lambda <lambda> <train_file>

To run peersim code:

1. run ./build_dsvm.sh to build the peersim code
2. modify config file within ./peersim-pegasos/config/
3. java -cp "lib/*:classes" -Djava.library.path=lib peersim.Simulator  <config_path>  data/"outputdata.txt"

** Note: All things inside the < > are supposed to be arguments you provide, and do not include "<" and ">" in the commands.


