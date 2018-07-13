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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.Arrays;
import java.lang.Integer;

import java.io.FileReader;

import java.io.LineNumberReader;

import jnipegasos.PrimalSVMWeights;

import peersim.config.Configuration;
import peersim.config.FastConfig;
import peersim.core.*;
import peersim.cdsim.*;

import jnipegasos.JNIPegasosInterface;
import jnipegasos.LearningParameter;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightInterface;
import java.net.MalformedURLException;
import java.text.ParseException;
import java.io.BufferedReader;

/**
 * Class GadgetProtocol
 * Implements a cycle based {@link CDProtocol}. It implements the Gadget algorithms
 * described in paper:
 * Chase Henzel, Haimonti Dutta
 * GADGET SVM: A Gossip-bAseD sub-GradiEnT SVM Solver   
 * 
 *  @author Deepak Nayak
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
	
	public static boolean optimizationDone = false;	

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
	
	private PrimalSVMWeights primalSVMWeights;
	
	private PrimalSVMWeights oldWeightVector;
	
	private double oldWeight;
	
	private boolean pushsum2_execute = true;
	
	private String protocol;


	
	private String resourcepath;
	
	private double peg_lambda;
	private int max_iter;
	private int exam_per_iter;
	

	/**
	 * Default constructor for configurable objects.
	 */
	public GadgetProtocol(String prefix) {
		lambda = Configuration.getDouble(prefix + "." + PAR_LAMBDA, 0.01);
		T = Configuration.getInt(prefix + "." + PAR_ITERATION, 100);
		//T = 0;
		lid = FastConfig.getLinkable(CommonState.getPid());
		primalSVMWeights = new PrimalSVMWeights(new TreeMap<Integer, Double>());
		oldWeightVector = new PrimalSVMWeights(new TreeMap<Integer, Double>());
		protocol = Configuration.getString(prefix + "." + "prot", "pushsum1");
		
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
	    System.out.println("Node "+node.getID()+" is gossiping with Node "+peer.getID()+"....");
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

	// Comment inherited from interface
	/**
	 * This is the method where actual algorithm is implemented. This method gets    
	 * called in each cycle for each node.
	 * NOTE: The Gadget algo's iteration corresponds to the inner loop, so call this 
	 * once only, i.e. keep simulation.cycles 1
	 */
	public void nextCycle(Node node, int pid) {
		
		//System.out.println("The number of GADGET iterations is " + T);
		
		int iter = CDState.getCycle();
		System.out.println("The current cycle is: " + CDState.getCycle());
		PegasosNode pn = (PegasosNode)node;
		
		resourcepath = pn.getResourcePath();
		peg_lambda = pn.getPegasosLambda();
		max_iter = pn.getMaxIter();
		exam_per_iter = pn.getExamPerIter();
		System.out.println("Norm: "+ pn.wtvector.getL2Norm() + " Node ID: " + pn.getID());
		


		
		//Check if the algorithm has converged or not.
		String lastLine = "";
		String objvaluefilename = pn.getResourcePath() + "/" + "m_" + pn.getID() + ".dat" + "_results.csv";
		try {
				
		String nextLine = "";
			BufferedReader br = new BufferedReader(new FileReader(objvaluefilename));
			while((nextLine=br.readLine())!=null) {
					lastLine = nextLine;
					
			}



		}
		catch(Exception e){}

		List<String> resultList = Arrays.asList(lastLine.split(","));

			int converged = Integer.parseInt(resultList.get(resultList.size() - 1));
		
		String trainfilename = pn.getResourcePath() + "/" + "t_" + pn.getID() + ".dat";
		String modelfilename = pn.getResourcePath() + "/" + "m_" + pn.getID() + ".dat";
		String testfilename = pn.getResourcePath() + "/" + "tst_" + pn.getID() + ".dat";
		//Record start time of cycle
				double startTime3=System.currentTimeMillis();

				pn.getTrainingData(trainfilename,pn.getID());		
				// Instantiate a pegasos interface
				JNIPegasosInterface trainer = new JNIPegasosInterface();
				// Set learning parameter here like lambda, max_iter, exam_per_iter here
				LearningParameter lp = new LearningParameter(peg_lambda, max_iter, exam_per_iter, 0, iter+1);

				System.out.println("Converged: " +  converged);
		if(converged == 0){

				
				
				// call the native training method. Actually not necessary for
				// GADGET, but just running to get local weights for comparing
			System.out.println("Training the model.");
				trainer.trainModel(trainfilename, modelfilename, testfilename, lp);
			}

			
		try {
			BufferedReader br = new BufferedReader(new FileReader(modelfilename));
			String nextLine = "";
			String output = "";
			while((nextLine=br.readLine())!=null) {
				String[] weights = nextLine.split(" ");
				for(String weight1: weights) {
					String[] w = weight1.split(":");
					if(w[1].equals("inf") || w[1].equals("-inf")) {
						// Do Nothing
					}
					else {
						output+=weight1;
						output+=" ";

					}
				}
			}
			output = output.trim();
			br.close();

			//Write the same weights back?
			BufferedWriter bw = new BufferedWriter(new FileWriter(modelfilename));
			bw.write(output);
			bw.close();
			double timeLocal = System.currentTimeMillis() - startTime3;
			//System.out.println("Time for local model construction  "+timeLocal/1000);
		}
		catch(Exception e) {
			
		}
		// Now, decode the training file and weight vector and save in vector form
		// for later use.
		try {
			// used this deprecated method to make svmlight call happy.
			pn.traindataset = SVMLightInterface.getLabeledFeatureVectorsFromURL(new File(trainfilename).toURL(), 0);
			pn.wtvector = trainer.getWeightsfromFile(modelfilename);
			//System.out.println("WeightVector before pushsum: ");
			//System.out.println(pn.wtvector);
			// try resetting here, so that simulation.cycle can be used and gossip
			// happens in more balanced way
			//result.wtvector.resetWeights();
			
			double weight = pn.wtvector.getL2Norm();
			//System.out.println("[init]: LOCAL weight norm, weight at node ["
            //                                    + pn.getID() + "]: "+ pn.wtvector.getL2Norm() + " ["+weight+"]");
		}
		catch (ParseException pe) {
			pe.printStackTrace();
			//System.out.println("parse error in " + trainfilename);
		}
		catch (MalformedURLException me) {
			me.printStackTrace();
		}
		
		pushsum1(node, pn, pid);

		//System.out.println("WeightVector after pushsum: ");
		//System.out.println(pn.wtvector);
		//Record start time of cycle
		
		double timeDist;
		TreeMap<Integer, Double> L = new TreeMap<Integer, Double>();
		
		System.out.println("current node ID: [" + pn.getID() + "]");
		
		int N = pn.traindataset.length;	// #data points
		double y;	// label
		

		// Classify and check
		//String predictfilename = "pred" + pn.getID() + ".dat";
		//trainer.classify(testfilename, modelfilename, predictfilename);
		

		//Project the weight and loss vectors
		Iterator<Integer> w_it = pn.wtvector.getWeights().keySet().iterator();
		Iterator<Integer> l_it = L.keySet().iterator();
		int count = 0;
		//Project the weight vector at this node
		while (w_it.hasNext()) 
			{
				Integer index = w_it.next();
				double scale = Math.min(1.0, 1.0 / (Math.sqrt(lambda) * pn.wtvector.getL2Norm()));
				// not sure if first term should be multiplied by N
				scale = scale * Math.sqrt(pn.getNumNodes());
				double newval = scale * pn.wtvector.getWeights().get(index);
				pn.wtvector.addFeature(index, newval);		
				
				
			} // end of w_it loop
		//timeDist = System.currentTimeMillis() - startTime;
		//double startTime2=System.currentTimeMillis();

		/*
		while (l_it.hasNext())
			 {
				Integer index = l_it.next();
				double lossterm = L.get(index);
				if(pn.wtvector.getWeights().containsKey(index)) 
				{
					pn.wtvector.addFeature(index, lossterm + 
							pn.wtvector.getWeights().get(index));
				//System.out.println("[DEBUG] Step 5");			
				}
				else 
				{
					pn.wtvector.addFeature(index,lossterm);
				}
				//timeDist = System.currentTimeMillis() - startTime;
				//System.out.println("Time inside this loop "+timeDist);
			} // end of l_it
		
		double timeDist2 = System.currentTimeMillis() - startTime2;
		*/
		//writeIntoFile(timeDist);
		//System.out.println("Time for running GADGET is "+timeDist/1000);
		//System.out.println("Time for loss updates is "+timeDist2/1000);
		//String timeFile = pn.getResourcePath() + "/" + "time_Vec_" + pn.getID() + ".csv";
		//writeWtVec(timeFile,timeDist);
		
				
		//System.out.println("[DEBUG] GADGET Norm of wt vector node[" + pn.getID() + "] :"+(pn.wtvector.getL2Norm())/N);
	 	//double nm=pn.wtvector.getL2Norm()/N;
	 	//String WtNormFile = pn.getResourcePath() + "/" + "Wt_Nm_" + pn.getID() + ".csv";		
		//writeWtVec(WtNormFile,pn.wtvector.getL2Norm()/N);	
		
		
		pn.misclassified = 0;	// reset the misclassified count in each iter
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
						//pn.misclassified++;
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
			
			if(Debug.ON) 
			{
				System.out.println("[DEBUG] #misclassified at node[" + pn.getID() + "] : "
							+ pn.misclassified);
				//String MisClassFile = pn.getResourcePath() + "/" + "Ms_Cl_" + pn.getID() + ".csv";		
				//writeWtVec(MisClassFile,pn.misclassified);	
			}
			
			
/*
System.out.println("Printing out global weights after this round of gossip.");
final int len = Network.size();
	for (int i = 0; i <  len; i++) {
		PegasosNode nodez = (PegasosNode) Network.get(i);
		nodez.writeGlobalWeights();
		
	
	}*/

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

	public static void writeIntoFile(String millis) {
		File file = new File("exec-time.txt");
		 
		// if file doesnt exists, then create it
		if (!file.exists()) {
			try {
				file.createNewFile();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		FileWriter fw;
		try {
			fw = new FileWriter(file.getAbsoluteFile(),true);

		BufferedWriter bw = new BufferedWriter(fw);
		bw.write(millis+"\n");
		bw.close();
		} catch (IOException e)
		 {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		

	}
	
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
	
	

}

