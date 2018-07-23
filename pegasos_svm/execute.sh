#!/bin/bash
# Basic until loop
counter=0
until [ $counter -gt 9 ]
do
./pegasos  -testFile ../../data/reuters/money-fx.tst -modelFile ../../data/reuters/pegasos_results/model_reuters.dat -iter 10000000 -lambda 1.29e-4 -round $counter ../../data/reuters/money-fx.trn
((counter++))
done
echo All done

 
