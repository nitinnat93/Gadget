#!/bin/bash
python train_gadget.py --dataset adult --reg_lambda 0.0000307 --max_iter 10000000000 --configfile config-pegasosAdult.cfg 
python train_gadget.py --dataset mnist --reg_lambda 0.0000167 --max_iter 10000000000 --configfile config-pegasosMnist.cfg 
python train_gadget.py --dataset cov2 --reg_lambda 0.000001 --max_iter 10000000000 --configfile config-pegasosCov.cfg
python train_gadget.py --dataset reuters --reg_lambda 0.0000129 --max_iter 1000000000 --configfile config-pegasosReuters.cfg
python train_gadget.py --dataset usps --reg_lambda 0.000136 --max_iter 1000000000 --configfile config-pegasosUsps.cfg 
python train_gadget.py --dataset ccat --reg_lambda 0.0001 --max_iter 1000000000 --configfile config-pegasosCCAT.cfg