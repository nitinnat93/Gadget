/*
 * Copyright (c) 2003-2005 The BISON Project
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 */

package peersim.gossip;

import java.util.Iterator;
import java.util.TreeMap;

import peersim.config.*;
import peersim.core.*;
import peersim.util.IncrementalStats;

/**
 * Print statistics for Push Sum computation. Statistics printed
 * are defined by {@link IncrementalStats#toString}
 * 
 * @author Raghuram Nagireddy
 */
public class PushSumObserver implements Control {

    // /////////////////////////////////////////////////////////////////////
    // Constants
    // /////////////////////////////////////////////////////////////////////

    /**
     * Config parameter that determines the accuracy for standard deviation
     * before stopping the simulation. If not defined, a negative value is used
     * which makes sure the observer does not stop the simulation
     * 
     * @config
     */
    private static final String PAR_ACCURACY = "accuracy";

    /**
     * The protocol to operate on.
     * 
     * @config
     */
    private static final String PAR_PROT = "protocol";

    // /////////////////////////////////////////////////////////////////////
    // Fields
    // /////////////////////////////////////////////////////////////////////

    /**
     * The name of this observer in the configuration. Initialized by the
     * constructor parameter.
     */
    private final String name;

    /**
     * Accuracy for standard deviation used to stop the simulation; obtained
     * from config property {@link #PAR_ACCURACY}.
     */
    private final double accuracy;
    
    private String protocol;


    // /////////////////////////////////////////////////////////////////////
    // Constructor
    // /////////////////////////////////////////////////////////////////////

    /**
     * Creates a new observer reading configuration parameters.
     */
    public PushSumObserver(String name) {
        this.name = name;
        accuracy = Configuration.getDouble(name + "." + PAR_ACCURACY, -1);
        protocol = Configuration.getString(name + "." + "prot", "pushsum1");
        
    }
    

    // /////////////////////////////////////////////////////////////////////
    // Methods
    // /////////////////////////////////////////////////////////////////////

    /**
     * Print statistics for a Push Sum computation. Statistics
     * printed are defined by {@link IncrementalStats#toString}. The current
     * timestamp is also printed as a first field.
     * 
     * @return if the standard deviation is less than the given
     *         {@value #PAR_ACCURACY}.
     */
    public boolean execute() {

    	if(GadgetProtocol.end) return true;
    	if(GadgetProtocol.pushsumobserverflag) return false;

       TreeMap<Integer, IncrementalStats> is1 = new TreeMap<Integer, IncrementalStats>();
       
        for (int i = 0; i < Network.size(); i++) {

        	Node n = Network.get(i);
        	PegasosNode pn1 = (PegasosNode) n;
        	
    		Iterator<Integer> p_it = pn1.wtvector.getWeights().keySet().iterator();
    		while (p_it.hasNext()) {
    			Integer index = p_it.next();
    			if(!is1.containsKey(index)) {
    				is1.put(index,new IncrementalStats());   				
    			}
    			if(protocol.equals("pushsum2"))
    				is1.get(index).add(pn1.wtvector.getWeights().get(index)/pn1.weight);
    			else
    				is1.get(index).add(pn1.wtvector.getWeights().get(index));
        	}
        }

        /* Terminate if accuracy target is reached */
        boolean retVal = true;
        Iterator<Integer> is_it =  is1.keySet().iterator();
        while (is_it.hasNext())  {
			Integer index = is_it.next();
        	retVal = retVal && (is1.get(index).getStD() <= accuracy);
        }
     
        if(retVal) {	// All weight components converged

        	System.out.println("Push-Sum converged...###########################!");
        	GadgetProtocol.pushsumobserverflag = true;
        	return GadgetProtocol.end;
        }
        else {
        	// Push-Sum not converged        	
        	return false;
        }
    }
}
