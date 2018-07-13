/*
 * Peersim-Gadget : A Gadget protocol implementation in peersim based on the paper
 * Chase Henzel, Haimonti Dutta
 * GADGET SVM: A Gossip-bAseD sub-GradiEnT SVM Solver   
 * 
 * Copyright (C) 2012
 * Deepak Nayak 
 * Columbia University, Computer Science MS'13
 * 
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
 */

package peersim.gossip;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.io.*;

import com.jamal.JamalException;
import com.jamal.MatlabCaller;
import com.jamal.client.MatlabClient;

import jnipegasos.PrimalSVMWeights;

import peersim.config.Configuration;
import peersim.config.FastConfig;
import peersim.core.*;
import peersim.cdsim.*;

/**
 * Class GadgetProtocol
 * Implements a cycle based {@link CDProcol}. It implements the Gadget algorithms
 * described in paper:
 * Chase Henzel, Haimonti Dutta
 * GADGET SVM: A Gossip-bAseD sub-GradiEnT SVM Solver   
 * 
 *  @author Raghuram Nagireddy
 */
public class GadgetProtocol implements CDProtocol {
	/**
	 * New config option to get the learning parameter lambda for GADGET
	 * @config
	 */
	private static final String PAR_LAMBDA = "lambda";
	/**
	 * New config option to get the number of iteration for GADGET
	 * @config
	 */
	private static final String PAR_ITERATION = "iter";
	
	public static boolean flag = false;
	
	public static int t = 0;
	
	private boolean optimizationDone = false;	

	/** Linkable identifier */
	protected int lid;
	/** Learning parameter for GADGET, different from lambda parameter in pegasos */
	protected double lambda;
	/** Number of iteration (T in gadget)*/
	protected int T;
	
	private int pushsumflag = 0;
	
	public static double[][] optimalB;
	
	public static boolean end = false;
	
	public static boolean pushsumobserverflag = false;
	
	private TreeMap<Integer, Double> primalSVMWeights;
	
	private TreeMap<Integer, Double> oldWeightVector;
		
	private double oldWeight;
	
	private boolean pushsum2_execute = true;
        //The number of iterations to which GADGET runs
        private int iter=0;
	private String protocol;
	private String method;
	private int parameter;	// The 'r' in new protocol
	

	/**
	 * Default constructor for configurable objects.
	 */
	public GadgetProtocol(String prefix) {
		lambda = Configuration.getDouble(prefix + "." + PAR_LAMBDA, 1000);
		T = Configuration.getInt(prefix + "." + PAR_ITERATION, 10);
		//T = 0;
		lid = FastConfig.getLinkable(CommonState.getPid());
		primalSVMWeights = new TreeMap<Integer, Double>();
		oldWeightVector = new TreeMap<Integer, Double>();

		protocol = Configuration.getString(prefix + "." + "prot", "pushsum1");
		method = Configuration.getString(prefix + "." + "method", "randomr");
		parameter = Configuration.getInt(prefix + "." + "param", 1);				
		
	}

	/**
	 * Returns true if it possible to deliver a response to the specified node,
	 * false otherwise.
	 * Currently only checking failstate, but later may be we need to check
	 * the non-zero transition probability also
	 */
	protected boolean canDeliverRequest(Node node) {
		if (node.getFailState() == Fallible.DEAD)
			return false;
		return true;
	}
	/**
	 * Returns true if it possible to deliver a response to the specified node,
	 * false otherwise.
	 * Currently only checking failstate, but later may be we need to check
	 * the non-zero transition probability also
	 */
	protected boolean canDeliverResponse(Node node) {
		if (node.getFailState() == Fallible.DEAD)
			return false;
		return true;
	}

	/**
	 * Clone an existing instance. The clone is considered 
	 * new, so it cannot participate in the aggregation protocol.
	 */
	public Object clone() {
		GadgetProtocol gp = null;
		try { gp = (GadgetProtocol)super.clone(); }
		catch( CloneNotSupportedException e ) {} // never happens
		return gp;
	}
	
	private void pushsum1(Node node, PegasosNode pn, int pid) {
		PegasosNode peer = (PegasosNode)selectNeighbor(node, pid);

		Iterator<Integer> p_it = peer.wtvector.getWeights().keySet().iterator();
		while (p_it.hasNext()) {
			// w and l are sorted
			Integer index = p_it.next();
			if(pn.wtvector.getWeights().containsKey(index)) {
				pn.wtvector.addFeature(index,  
						(peer.wtvector.getWeights().get(index) +
						pn.wtvector.getWeights().get(index))/2);
			}
			else {
				pn.wtvector.addFeature(index, 
						peer.wtvector.getWeights().get(index)/2);
			}
			peer.wtvector.addFeature(index, pn.wtvector.getWeights().get(index));				
			
		} // push sum done

	}
	protected List<Node> getPeers(Node node) {
		Linkable linkable = (Linkable) node.getProtocol(lid);
		if (linkable.degree() > 0) {
			List<Node> l = new ArrayList<Node>(linkable.degree());			
			for(int i=0;i<linkable.degree();i++) {
				l.add(linkable.getNeighbor(i));
			}
			return l;
		}
		else
			return null;						
	}	
	
	private void pushsum2(Node node, PegasosNode pn, int pid) {

		if(!pushsum2_execute) {

			Iterator<Integer> p_it = oldWeightVector.keySet().iterator();

			while (p_it.hasNext()) {
				Integer index = p_it.next();
				pn.wtvector.addFeature(index,oldWeightVector.get(index));
			}
			pn.weight = oldWeight;
			pushsum2_execute = !pushsum2_execute;		
			return;
		}

		oldWeightVector = new TreeMap<Integer, Double>();

		Iterator<Integer> p_it1 = pn.wtvector.getWeights().keySet().iterator();
		while (p_it1.hasNext()) {
			// w and l are sorted
			Integer index = p_it1.next();
			oldWeightVector.put(index, optimalB[(int)node.getID()][(int)node.getID()]*pn.wtvector.getWeights().get(index));
		}
		oldWeight = optimalB[(int)node.getID()][(int)node.getID()]*pn.weight;

		List<Node> peers = getPeers(node);
		for(Node peer1:peers) {
			PegasosNode peer = (PegasosNode)peer1;
			Iterator<Integer> p_it = peer.wtvector.getWeights().keySet().iterator();
			while (p_it.hasNext()) {
				// w and l are sorted
				Integer index = p_it.next();
				if(oldWeightVector.containsKey(index)) {
					oldWeightVector.put(index,  
							optimalB[(int)peer.getID()][(int)node.getID()]*peer.wtvector.getWeights().get(index) +
							oldWeightVector.get(index));
				}
				else {
					oldWeightVector.put(index, 
							optimalB[(int)peer.getID()][(int)node.getID()]*peer.wtvector.getWeights().get(index));
				}
			}
			oldWeight += optimalB[(int)peer.getID()][(int)node.getID()]*peer.weight;	
			
		}// push sum done
		pushsum2_execute = !pushsum2_execute;

	}	
	
	/**
	 * Some utility functions (not used currently)
	 * @param array
	 * @param n
	 * @return
	 */
	private double[][] toMatrix(double[] array, int n) {
		double[][] mat = new double[n][n];
		for(int i=0;i<n;i++) {
			for(int j=0;j<n;j++) {
				mat[i][j] = array[i*n+j];
			}
		}
		return mat;
	}
	
	private double[][] getAdjMat() {
		int networkSize = Network.size();
		double[][] adjM = new double[networkSize][networkSize];
		for(int i=0;i<networkSize;i++) {
			Node n = Network.get(i);
			Linkable l = (Linkable) n.getProtocol(lid);
			for(int j=0;j<l.degree();j++) {
				Node ne = l.getNeighbor(j);
				adjM[(int)n.getID()][(int)ne.getID()] = 1;
			}			
		}
		return adjM;
	}
	
	/**
	 * Initially used to implement SDP formulation but is now also used for the new protocol
	 * @param node
	 */
	private void generateOptimalB(Node node) {
		int networkSize = Network.size();
		if(optimalB == null)
			optimalB = new double[networkSize][networkSize];
		
		if("randomr".equals(method)) {	// The new protocol case
			selectRandomNeighbors(node, parameter);
			return;
		}
		double[][] adjM = getAdjMat();
        try {
            MatlabClient matlabClient = new MatlabClient(
                    MatlabCaller.HOST_ADDRESS,
                    "/home/raghuram/MATLAB/R2012a/bin/matlab.exe",40);
            Object[] inArgs = new Object[1];
            inArgs[0] = adjM;
            Object[] outputArgs = matlabClient.executeMatlabFunction("fmmc",
                    inArgs, 2);
            double[] prob = (double[])outputArgs[0]; 
            optimalB = toMatrix(prob,networkSize);            
            matlabClient.shutDownServer();
 
        } catch (JamalException e) {
            e.printStackTrace();
        } catch (Exception e) {
        	e.printStackTrace();
        }
	}
	
	/**
	 * Uitility function to print B matrix
	 * @param mat
	 * @param row
	 */
	private void printMatrix(double[][] mat,int row) {
		int networkSize = Network.size();		
		for(int i=0;i<networkSize;i++) {
			for(int j=0;j<networkSize;j++) {			
				System.out.print(mat[i][j]+" ");
			}
			System.out.println(";");
		}
	}
	
    /**
       Utility function to print a vector to a file.
       Used to print the time taken by an iteration of nextCycle()
     */
       public void writeWtVec(String fName, double db)
       {
	BufferedWriter out = null;
	try
	    {
		FileWriter fstream = new FileWriter(fName, true);
		out = new BufferedWriter(fstream);
		//IlpVector vLoss = new IlpVector(ls);
		out.write(String.valueOf(db)); 
		out.write("\n");
	    }
	catch (IOException ioe) 
	    {ioe.printStackTrace();}
	finally
	    {
		if (out != null) 
		    {try {out.close();} catch (IOException e) {e.printStackTrace();}
		    }
	    }
        }

	// Comment inherited from interface
	/**
	 * This is the method where actual algorithm is implemented. This method gets 
	 * called in each cycle for each node.
	 * NOTE: The Gadget algo's iteration corresponds to the inner loop, so call this 
	 * once only, i.e. keep simulation.cycles 1 --> This does not seem to work!!!
	 */
	public void nextCycle(Node node, int pid) {
	        
	        //Get the cycle number
	        int iterNum = CDState.getCycle()+1;
	        System.out.println("Cycle number is "+CDState.getCycle());
		//while (iterNum < T)
		//{
                int resetflagto = pushsumflag;

		PegasosNode pn = (PegasosNode)node;

		//Record start time of cycle
		double startTime=System.currentTimeMillis();

		if(node.getID()==0 && pushsumflag == 0)	{t=t+1;}
		System.out.println("iterNum is: "+iterNum);
                System.out.println("T is: "+T);
		if(protocol.equals("pushsum2")) {
			if(!optimizationDone) {	// Only optimize in the first iteration
				generateOptimalB(node);
				optimizationDone = true;
				return;
			}		
		}

		if(t>T) {
			end = true;			
			return;
		}
		else if(pushsumflag == 1 && !pushsumobserverflag) {
			if(protocol.equals("pushsum2")) {
				generateOptimalB(node);	// Choose random neighbors in every cycle
				pushsum2(node, pn, pid);
			}
			else
			    {	
			     pushsum1(node, pn, pid);
			     System.out.println("[DEBUG] #GADGET norm is "+pn.wtvector.getL2Norm());
			    }
			return;
		}
		else if(pushsumflag == 1) {	// Push-Sum converged

			double scale = Math.min(1.0, 1.0 / (Math.sqrt(lambda) * pn.wtvector.getL2Norm()));
			for (Map.Entry<Integer, Double> entry : pn.wtvector.getWeights().entrySet()) {
				pn.wtvector.addFeature(entry.getKey(), scale * entry.getValue());
				if(primalSVMWeights.containsKey(entry.getKey())) {
					primalSVMWeights.put(entry.getKey(), primalSVMWeights.get(entry.getKey())+scale * entry.getValue());
				}
				else {
					primalSVMWeights.put(entry.getKey(), scale * entry.getValue());					
				}			
			}
			resetflagto = 0;		
			if(protocol.equals("pushsum2")) {
				pushsum2_execute = true;
			}
			
		}
		else if(pushsumflag == 0) {	
			pn.weight = 1;				
	
			TreeMap<Integer, Double> L = new TreeMap<Integer, Double>();
			
			int N = pn.traindataset.length;	// #data points
			double y;	// label
			if(flag==false) {
				flag = true;
			}
			pn.misclassified = 0;	// reset the misclassified count in each iter
			System.out.println("[DEBUG]: Gadget weight norm is "+pn.wtvector.getL2Norm());
			for (int n = 0; n < N; n++) { // data point loop
				y = pn.traindataset[n].getLabel();
				int xsize = pn.traindataset[n].size();
				double dotprod = 0.0;

				// calculate <w,x> using two iterator which moves over x[i] and w
				for (int xiter = 0; xiter < xsize; xiter++) { // dot product loop
					int xdim = pn.traindataset[n].getDimAt(xiter);
					double xval = pn.traindataset[n].getValueAt(xiter);
					if(pn.wtvector.getWeights().containsKey(xdim)) {// wtvector has this dim
						double wval = pn.wtvector.getWeights().get(xdim);
						dotprod += xval * wval;
					}

				}// dot product loop end
				if ((y * dotprod) < 1) { // this point is in Si+
					if((y * dotprod) < 0) pn.misclassified++;
					// Li calculated.
					for(int xiter = 0; xiter < xsize; xiter++) {// xsize loop
						int xkey = pn.traindataset[n].getDimAt(xiter);
						double xval = pn.traindataset[n].getValueAt(xiter);
						if(L.containsKey(xkey)) {
							L.put(xkey, L.get(xkey) + y * xval);
						}
						else
							L.put(xkey, y * xval);
					}//xsize loop
				}

			} // data point loop end
			if(Debug.ON) {
				System.out.println("[DEBUG] #misclassified at node[" + pn.getID() + "] : "
							+ pn.misclassified);
			}
			double alpha = 1.0 / (lambda * t); // our loop starts from 0
			//calculate w_t1/2, what is ni??
			Iterator<Integer> w_it = pn.wtvector.getWeights().keySet().iterator();
			Iterator<Integer> l_it = L.keySet().iterator();
			// Lots of confusion, so do it in two step
			// inefficient but clean
			while (w_it.hasNext()) {
				Integer index = w_it.next();
				// not sure if first term should be multiplied by N
				double newval = (1 - lambda * alpha) * N * pn.wtvector.getWeights().get(index);
				pn.wtvector.addFeature(index, newval);		
			}
			while (l_it.hasNext()) {
				Integer index = l_it.next();
				double lossterm = L.get(index);
				if(pn.wtvector.getWeights().containsKey(index)) {
					pn.wtvector.addFeature(index, alpha * lossterm + 
							pn.wtvector.getWeights().get(index));
				}
				else {
					pn.wtvector.addFeature(index, alpha * lossterm);
				}
			} // ~w_t1/2 calculated, now do push sum
			
			resetflagto = 1;
			pushsumobserverflag = false;

		}
		pushsumflag = resetflagto;
		String timeFile = pn.getResourcePath() + "/" + "time_Vec_" + pn.getID() + ".csv";
		double timeDist = System.currentTimeMillis() - startTime;
		writeWtVec(timeFile,timeDist);
		System.out.println("Time for running GADGET "+timeDist);
		//}  end of while loop for GADGET iteration
	}

	/**
	 * Selects a random neighbor from those stored in the {@link Linkable} protocol
	 * used by this protocol.
	 */
	protected Node selectNeighbor(Node node, int pid) {
		Linkable linkable = (Linkable) node.getProtocol(lid);
		if (linkable.degree() > 0) 
			return linkable.getNeighbor(
					CommonState.r.nextInt(linkable.degree()));
		else
			return null;
	}


	// Choosing some 'r' neighbors randomly
	protected void selectRandomNeighbors(Node node, int r) {
		Linkable linkable = (Linkable) node.getProtocol(lid);
		if (linkable.degree() > 0) {
			List<Long> l = new ArrayList<Long>();
			for(int i=0;i<linkable.degree();i++)
				l.add(linkable.getNeighbor(i).getID());
			Collections.shuffle(l);
			for(int i=0;i<l.size();i++)
				optimalB[(int)node.getID()][l.get(i).intValue()] = 0;
			optimalB[(int)node.getID()][(int)node.getID()] = (double)1/(r+1);
			for(int i=0;i<r;i++)
				optimalB[(int)node.getID()][l.get(i).intValue()] = (double)1/(r+1);
		}
		
	}	

}
