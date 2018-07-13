for i in 15 30 45 60 75 90 105 120
do
echo "random.seed 1234567890 
# keep simulation.cycle 1 
# number of iteration is controlled by GadgetProtocol.iter 
simulation.cycles 8000 

network.size $i
network.node.size $i 
network.node peersim.gossip.PegasosNode 
network.node.resourcepath /home/raghuram/Downloads/data_covtype 
# These three parameter is used by pegasos, if not given 
# takes default values 0.001, 100000 and 1 
network.node.lambda 0.01   
network.node.maxiter 1000   
network.node.examperiter 1000  

# connectivity of nodes 
degree $((i-5))

protocol.0 peersim.core.IdleProtocol  
protocol.0.cache degree  

protocol.1 peersim.gossip.GadgetProtocol4  
protocol.1.linkable 0  
# learning rate and iter for GADGET, keep lambda smaller and iter larger 
protocol.1.lambda 0.01  
protocol.1.iter 20  
protocol.1.prot pushsum1
protocol.1.method randomr  
protocol.1.param 1


init.0 peersim.dynamics.WireRingLattice  
init.0.protocol 0  
init.0.k degree  

control.d0 peersim.gossip.PushSumObserver  
control.d0.protocol 1  
control.d0.accuracy 0.01  
control.d0.prot pushsum1 



# final control only runs once at last, so any cleanup can be done here 
control.f0 peersim.gossip.FinalControl 
control.f0.protocol 1 
control.f0.until 0 
control.f0.step 1 
control.f0.FINAL" > config/auto.cfg
java -cp lib/*:classes -Djava.library.path=lib peersim.Simulator config/auto.cfg 2> out_data_covtype/$i"_uniform.txt"
for p in $(seq 1 $((i-5)))
do
echo "random.seed 1234567890 
# keep simulation.cycle 1 
# number of iteration is controlled by GadgetProtocol.iter 
simulation.cycles 8000 

network.size $i
network.node.size $i 
network.node peersim.gossip.PegasosNode 
network.node.resourcepath /home/raghuram/Downloads/data_covtype 
# These three parameter is used by pegasos, if not given 
# takes default values 0.001, 100000 and 1 
network.node.lambda 0.01   
network.node.maxiter 1000   
network.node.examperiter 1000  

# connectivity of nodes 
degree $((i-5))

protocol.0 peersim.core.IdleProtocol  
protocol.0.cache degree  

protocol.1 peersim.gossip.GadgetProtocol4  
protocol.1.linkable 0  
# learning rate and iter for GADGET, keep lambda smaller and iter larger 
protocol.1.lambda 0.01  
protocol.1.iter 20  
protocol.1.prot pushsum2
protocol.1.method randomr  
protocol.1.param $p 


init.0 peersim.dynamics.WireRingLattice  
init.0.protocol 0  
init.0.k degree  

control.d0 peersim.gossip.PushSumObserver  
control.d0.protocol 1  
control.d0.accuracy 0.01  
control.d0.prot pushsum2 



# final control only runs once at last, so any cleanup can be done here 
control.f0 peersim.gossip.FinalControl 
control.f0.protocol 1 
control.f0.until 0 
control.f0.step 1 
control.f0.FINAL" > config/auto.cfg
java -cp lib/*:classes -Djava.library.path=lib peersim.Simulator config/auto.cfg 2> out_data_covtype/$i"_"$p".txt"
done
done
