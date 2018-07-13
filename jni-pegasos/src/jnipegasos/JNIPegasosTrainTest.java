
import java.io.File;
import java.net.URL;

import jnipegasos.JNIPegasosInterface;
import jnipegasos.LearningParameter;
import jnipegasos.PrimalSVMWeights;

import com.martiansoftware.jsap.*;
import com.martiansoftware.jsap.stringparsers.*;

public class JNIPegasosTrainTest {
		
	public static void main(String[] args) throws Exception {
		// Commandline parsing, get the model file name and train datafile name
		// in commandline
		JSAP jsap = new JSAP();
		// longFlag datafile, longFlag modelfile
		FlaggedOption opt1 = new FlaggedOption("data")
									.setLongFlag("dataFile")
									.setRequired(true)
									.setStringParser(JSAP.STRING_PARSER);
		jsap.registerParameter(opt1);
		FlaggedOption opt2 = new FlaggedOption("model")
									.setLongFlag("modelFile")
									.setDefault("noModelFile")
									.setRequired(true)
									.setStringParser(JSAP.STRING_PARSER);
		jsap.registerParameter(opt2);

		FlaggedOption opt3 = new FlaggedOption("test")
									.setLongFlag("testFile")
									.setDefault("noTestFile")
									.setRequired(true)
									.setStringParser(JSAP.STRING_PARSER);
		jsap.registerParameter(opt3);
		
		JSAPResult config = jsap.parse(args);
		String dataFile = config.getString("data");
		String modelFile = config.getString("model");
		String testFile = config.getString("test");
		// Instantiate a pegasos interface
		JNIPegasosInterface trainer = new JNIPegasosInterface();
		// Set learning parameter here like lambda, max_iter, exam_per_iter here
PrimalSVMWeights wts = new PrimalSVMWeights();	

wts = trainer.getWeightsfromFile(modelFile); 	
System.out.println("Weight norm before training: " + wts.getL2Norm());
LearningParameter lp = new LearningParameter(1.29e-05, 1000, 1, 1);
		// call the native training method 
		trainer.trainModel(dataFile, modelFile, testFile,lp);
wts = trainer.getWeightsfromFile(modelFile); 
System.out.println("Weight norm after training: " + wts.getL2Norm());

lp.setReplace(0);

wts = trainer.getWeightsfromFile(modelFile); 
System.out.println("Weight norm before training: " + wts.getL2Norm());
trainer.trainModel(dataFile, modelFile, testFile,lp);
wts = trainer.getWeightsfromFile(modelFile); 
System.out.println("Weight norm after training: " + wts.getL2Norm());


wts = trainer.getWeightsfromFile(modelFile); 
System.out.println("Weight norm before training: " + wts.getL2Norm());
trainer.trainModel(dataFile, modelFile, testFile,lp);
wts = trainer.getWeightsfromFile(modelFile); 
System.out.println("Weight norm after training: " + wts.getL2Norm());

wts = trainer.getWeightsfromFile(modelFile); 
System.out.println("Weight norm before training: " + wts.getL2Norm());
trainer.trainModel(dataFile, modelFile, testFile,lp);
wts = trainer.getWeightsfromFile(modelFile); 
System.out.println("Weight norm after training: " + wts.getL2Norm());

wts = trainer.getWeightsfromFile(modelFile); 
System.out.println("Weight norm before training: " + wts.getL2Norm());
trainer.trainModel(dataFile, modelFile, testFile,lp);
wts = trainer.getWeightsfromFile(modelFile); 
System.out.println("Weight norm after training: " + wts.getL2Norm());

//trainer.trainModel(dataFile, modelFile, testFile,lp);
//trainer.trainModel(dataFile, modelFile, testFile,lp);
//trainer.trainModel(dataFile, modelFile, testFile,lp);
//trainer.trainModel(dataFile, modelFile, testFile,lp);
//trainer.trainModel(dataFile, modelFile, testFile,lp);
//trainer.trainModel(dataFile, modelFile, testFile,lp);
//trainer.trainModel(dataFile, modelFile, testFile,lp);
//trainer.trainModel(dataFile, modelFile, testFile,lp);
//trainer.trainModel(dataFile, modelFile, testFile,lp);
//trainer.trainModel(dataFile, modelFile, testFile,lp);
//trainer.trainModel(dataFile, modelFile, testFile,lp);
//trainer.trainModel(dataFile, modelFile, testFile,lp);
//trainer.trainModel(dataFile, modelFile, testFile,lp);		
	}
}
