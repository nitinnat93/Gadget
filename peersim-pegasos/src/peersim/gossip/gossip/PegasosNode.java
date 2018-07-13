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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.File;
import java.io.IOException;
import java.io.LineNumberReader;
import java.net.MalformedURLException;
import java.text.ParseException;
import java.util.Map;
import java.util.TreeMap;

import peersim.config.*;
import peersim.core.*;

import jnipegasos.JNIPegasosInterface;
import jnipegasos.LearningParameter;
import jnipegasos.PrimalSVMWeights;
import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightInterface;

/**
 * Class PegasosNode
 * An implementation of {@link Node} which can handle external resources. 
 * It is based on {@link GeneralNode} and extended it to be used with pegasos solver.
 * At the start of node, each node has some associated data file, on which it calls
 * training function through a jni call. 
 * It will take the resourcepath from config file.
 * This is another implementation of {@link Node} class that is used to compose the
 * p2p {@link Network}, where each node can handle an external resource.
 * @author Deepak Nayak
 */
public class PegasosNode implements Node {

	// ================= fields ========================================
	// =================================================================

	/**
	 * New config option added to get the resourcepath where the resource file
	 * should be generated. Resource files are named <ID> in resourcepath.
	 * @config
	 */
	private static final String PAR_PATH = "resourcepath";
	
	private static final String PAR_SIZE = "size";


	/**
	 * New config options added to set the learning parameters of pegasos
	 * PAR_LAMBDA 	: lambda parameter, default value 0.001 
	 * PAR_MAXITER 	: maximum number of iteration, defaults to 100000
	 * PAR_EXAM_PER_ITER 	: number of examples to consider for stochastic
	 * gradient computation, defaults to 1
	 * @config
	 */
	private static final String PAR_LAMBDA = "lambda";
	private static final String PAR_MAXITER = "maxiter";
	private static final String PAR_EXAM_PER_ITER = "examperiter";
	private static final String PAR_REPLACE = "replace";
        //private static final String PAR_ITER= "iter"; 

	/** used to generate unique IDs */
	private static long counterID = -1;

	/**
	 * The protocols on this node.
	 */
	protected Protocol[] protocol = null;

	/**
 	 * learning parameters of pegasos
 	 */
	private double lambda;
	private int max_iter;
	private int exam_per_iter;
        private int iter;
        private int replace;
	
        /**
	 * The current index of this node in the node
	 * list of the {@link Network}. It can change any time.
	 * This is necessary to allow
	 * the implementation of efficient graph algorithms.
	 */
	private int index;

	/**
	 * The fail state of the node.
	 */
	protected int failstate = Fallible.OK;

	/**
	 * The ID of the node. It should be final, however it can't be final because
	 * clone must be able to set it.
	 */
	private long ID;

	/**
	 * The prefix for the resources file. All the resources file will be in prefix 
	 * directory. later it should be taken from configuration file.
	 */
	private String resourcepath;

	/**
	 * The training dataset
	 */
	public LabeledFeatureVector[] traindataset;

	/**
	 * The primal weight vector
	 */
	public PrimalSVMWeights wtvector;
	
	public double weight;
	
	/** Misclassification count for debugging*/
	public int misclassified;
	public double obj_value;
	
	private int numNodes;

	// ================ constructor and initialization =================
	// =================================================================

	/** Used to construct the prototype node. This class currently does not
	 * have specific configuration parameters and so the parameter
	 * <code>prefix</code> is not used. It reads the protocol components
	 * (components that have type {@value peersim.core.Node#PAR_PROT}) from
	 * the configuration.
	 */

	
	public PegasosNode(String prefix) {
		System.out.println(prefix);
		String[] names = Configuration.getNames(PAR_PROT);
		resourcepath = (String)Configuration.getString(prefix + "." + PAR_PATH);
		lambda = Configuration.getDouble(prefix + "." + PAR_LAMBDA, 0.001);
		max_iter = Configuration.getInt(prefix + "." + PAR_MAXITER, 100000);
		exam_per_iter = Configuration.getInt(prefix + "." + PAR_EXAM_PER_ITER, 1);
		replace = Configuration.getInt(prefix + "." + PAR_REPLACE, 1);
		//iter = Configuration.getInt(prefix + "." + PAR_ITER);
		System.out.println("model file and train file are saved in: " + resourcepath);
		CommonState.setNode(this);
		ID = nextID();
		protocol = new Protocol[names.length];
		for (int i=0; i < names.length; i++) {
			CommonState.setPid(i);
			Protocol p = (Protocol) 
					Configuration.getInstance(names[i]);
			protocol[i] = p; 
		}
		numNodes = Configuration.getInt(prefix + "." + PAR_SIZE, 20);
		System.out.println("Number of nodes is ####### "+numNodes);
	}
	
	
	public void getTrainingData(String trainfilename,long id) {
		
		int id1 = (int)id;
//		int numTrainingPoints = 581000/numNodes;
		int numTrainingPoints = 7291/numNodes;	// This needs to be changed according to the dataset size

	    int start = id1*numTrainingPoints+1;
	    int end = start+numTrainingPoints-1;
//		String traindata = resourcepath + "/" + "covtype";
		String traindata = resourcepath + "/" + "zip.trn"; // This needs to be changed according to the dataset

	    
	    try {
		    final LineNumberReader in = new LineNumberReader(new FileReader(traindata));
		    String line=null;
			File file = new File(trainfilename);
			 
			// if file doesnt exist, then create it
			if (!file.exists()) {
				file.createNewFile();
			}
			else {
				file.delete();
				file.createNewFile();				
			}
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);		
		    while ((line = in.readLine()) != null && in.getLineNumber() <= end) {
		        if (in.getLineNumber() >= start) {	        	
					bw.write(line+"\n");
		            //System.out.println(line);
		        }
		    }
			bw.close();
			in.close();
	    
		}
	    catch(Exception e) {
			System.out.println("getTrainingData");
	    	
	    }
	    
	}

	/**
	 * Used to create actual Node by calling clone() on a prototype node. So, actually 
	 * a Node constructor is only called once to create a prototype node and after that
	 * all nodes are created by cloning it.
	 */

	public Object clone() {
		PegasosNode result = null;
		
		
		try { result=(PegasosNode)super.clone(); }
		catch( CloneNotSupportedException e ) {} // never happens
		result.protocol = new Protocol[protocol.length];
		CommonState.setNode(result);
		result.ID = nextID();
		for(int i=0; i<protocol.length; ++i) {
			CommonState.setPid(i);
			result.protocol[i] = (Protocol)protocol[i].clone();
		}
		System.out.println("creating node with ID: " + result.getID());
		// take the training datafile associated with it and call training function
		// and store the result locally in model file
		// currently training file name format is fixed and hardcoded, should be 
		// changed in future
		
		String trainfilename = resourcepath + "/" + "t_" + result.getID() + ".dat";
		String modelfilename = resourcepath + "/" + "m_" + result.getID() + ".dat";
		String testfilename = resourcepath + "/" + "tst_" + result.getID() + ".dat";
		//Record start time of cycle
		double startTime=System.currentTimeMillis();

		getTrainingData(trainfilename,result.getID());		
		// Instantiate a pegasos interface
		JNIPegasosInterface trainer = new JNIPegasosInterface();
		// Set learning parameter here like lambda, max_iter, exam_per_iter here
		LearningParameter lp = new LearningParameter(lambda, max_iter, exam_per_iter, 1, 0);
		// call the native training method. Actually not necessary for
		// GADGET, but just running to get local weights for comparing
		
		trainer.trainModel(trainfilename, modelfilename, testfilename, lp);
		


		//read weights from modelfile
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
			double timeLocal = System.currentTimeMillis() - startTime;
			//System.out.println("Time for local model construction  "+timeLocal/1000);
		}
		catch(Exception e) {
			
		}
		
		// Now, decode the training file and weight vector and save in vector form
		// for later use.
		try {
			// used this deprecated method to make svmlight call happy.
			result.traindataset = SVMLightInterface.getLabeledFeatureVectorsFromURL(new File(trainfilename).toURL(), 0);
			result.wtvector = trainer.getWeightsfromFile(modelfilename); 
			//System.out.println("WeightVector: ");
			//System.out.println(result.wtvector);
			// try resetting here, so that simulation.cycle can be used and gossip
			// happens in more balanced way
			//result.wtvector.resetWeights();
			weight = result.traindataset.length; 

			//System.out.println("[init]: LOCAL weight norm, weight at node ["
            //                                    + result.getID() + "]: "+ result.wtvector.getL2Norm() + " ["+weight+"]");
		}
		catch (ParseException pe) {
			pe.printStackTrace();
			//System.out.println("parse error in " + trainfilename);
		}
		catch (MalformedURLException me) {
			me.printStackTrace();
		}

		// Classify and check
		String predictfilename = "pred" + result.getID() + ".dat";
		trainer.classify(testfilename, modelfilename, predictfilename);
		return result;
		

	}

	/**
	 * This method is called at the end of simulation cycles. And it writes final 
	 * global weights obtained to global_<id>.dat files in resourcepath
	 */
	public void writeGlobalWeights() {
		String filename = resourcepath + "/" + "global_" + this.getID() + ".dat";
		BufferedWriter out = null;
		try {
			FileWriter fstream = new FileWriter(filename);
			out = new BufferedWriter(fstream);
			TreeMap<Integer, Double> map = this.wtvector.getWeights();
			System.out.println("[finish]: global weight norm at node["
                                                + this.getID() + "]: "+ this.wtvector.getL2Norm());
			for (Map.Entry<Integer, Double> entry : map.entrySet()) {
				String buf = entry.getKey().toString() + ":" + entry.getValue().toString() + " ";
				out.write(buf);
			}
		}
		catch (IOException ioe) {
			ioe.printStackTrace();
		}
		finally {
			if (out != null) {
				try {out.close();}
				catch (IOException ioe) {
					ioe.printStackTrace();
				}
			}
		}
	}

	/** returns the next unique ID */
	private long nextID() {

		return counterID++;
	}

	// =============== public methods ==================================
	// =================================================================


	public void setFailState(int failState) {

		// after a node is dead, all operations on it are errors by definition
		if(failstate==DEAD && failState!=DEAD) throw new IllegalStateException(
				"Cannot change fail state: node is already DEAD");
		switch(failState)
		{
		case OK:
			failstate=OK;
			break;
		case DEAD:
			//protocol = null;
			index = -1;
			failstate = DEAD;
			for(int i=0;i<protocol.length;++i)
				if(protocol[i] instanceof Cleanable)
					((Cleanable)protocol[i]).onKill();
			break;
		case DOWN:
			failstate = DOWN;
			break;
		default:
			throw new IllegalArgumentException(
					"failState="+failState);
		}
	}

	public int getFailState() { return failstate; }

	public boolean isUp() { return failstate==OK; }

	public Protocol getProtocol(int i) { return protocol[i]; }

	public int protocolSize() { return protocol.length; }

	public int getIndex() { return index; }

	public void setIndex(int index) { this.index = index; }
        
        public String getResourcePath(){return resourcepath;}
        public double getPegasosLambda(){return lambda;}
        public int getMaxIter(){return max_iter;}
        public int getExamPerIter(){ return exam_per_iter;}
        public int getReplace(){ return replace;}
        public int getNumNodes(){ return numNodes;}
	/**
	 * Returns the ID of this node. The IDs are generated using a counter
	 * (i.e. they are not random).
	 */
	public long getID() { return ID; }

	public String toString() 
	{
		StringBuffer buffer = new StringBuffer();
		buffer.append("ID: "+ID+" index: "+index+"\n");
		for(int i=0; i<protocol.length; ++i)
		{
			buffer.append("protocol[" + i +"]=" + protocol[i] + "\n");
		}
		return buffer.toString();
	}

	/** Implemented as <code>(int)getID()</code>. */
	public int hashCode() { return (int)getID(); }

}
