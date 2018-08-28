/** \file pegasos_jni.cc
 * Implementation of JNI methods
 * This file implements the native methods which is auto generated by jni
 * in jnipegasos_JNIPegasosInterface.h from JNIPegasosInterface.java. Internally
 * it calls the methods in Pegasos_optimize and other pegasos svm files.
 * 
 * @author Deepak Nayak
 * copyright GNU Public License 
 */
#include "pegasos_jni.h"
#include <string>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "pegasos_optimize.h"
#include "jnipegasos_JNIPegasosInterface.h"
using namespace std;

vector<float> GetLastLine(string filename){
// Reads the last line of the file and returns thr elements in a float array.
string lastLine;
ifstream fin;
fin.open(filename);
    if(fin.is_open()) {
      // Have to alter convergence num accordingly.
// Check the last line of the file.

        fin.seekg(-1,ios_base::end);                // go to one spot before the EOF

        bool keepLooping = true;
        while(keepLooping) {
            char ch;
            fin.get(ch);                            // Get current byte's data

            if((int)fin.tellg() <= 1) {             // If the data was at or before the 0th byte
                fin.seekg(0);                       // The first line is the last line
                keepLooping = false;                // So stop there
            }
            else if(ch == '\n') {                   // If the data was a newline
                keepLooping = false;                // Stop at the current position.
            }
            else {                                  // If the data was neither a newline nor at the 0 byte
                fin.seekg(-2,ios_base::cur);        // Move to the front of that data, then to the front of the data before it
            }
        }

                  
        getline(fin,lastLine);                      // Read the current line
            // Display it

        fin.close();
      }
      else{ cout << "Could not open the file.";}

      // Split the string into parts and store in an array
      
      vector<float> objects;
      string a;
      for(stringstream sst(lastLine); getline(sst, a, ','); )  // that's all ! 
       objects.push_back(stof(a)); //convert to float and push into float array

      return objects;

}

JNIEXPORT void JNICALL Java_jnipegasos_JNIPegasosInterface_trainmodel
(JNIEnv *env, jobject obj, jstring datafilename, jstring modelfilename, jstring testfilename, jobject lp) {
  // get filenames from arguments by converting jstring to native string
  const char* cdatafilename = env->GetStringUTFChars(datafilename, 0);
  const char* cmodelfilename = env->GetStringUTFChars(modelfilename, 0);
  const char* ctestfilename = env->GetStringUTFChars(testfilename, 0);
  //std::string testFile = "noTestFile";
  std::string testFile(ctestfilename);
  std::string modelFile(cmodelfilename);
  std::string dataFile(cdatafilename);
  env->ReleaseStringUTFChars(datafilename, cdatafilename);
  env->ReleaseStringUTFChars(modelfilename, cmodelfilename);

  // read datafile and initialize vectors
  uint dimension = 0;
  std::vector<simple_sparse_vector> Dataset;
  std::vector<int> Labels;
  double readingTime;
  ReadData(dataFile,Dataset,Labels,dimension,readingTime);

  uint testDimension = 0;
  std::vector<simple_sparse_vector> testDataset;
  std::vector<int> testLabels;
  double testReadingTime;
  if (testFile != "noTestFile") {
    ReadData(testFile,testDataset,testLabels,testDimension,testReadingTime);
  } else {
    testReadingTime = 0;
  }
    
  // these variables will be filled by training method
  double trainTime;
  double calc_obj_time;
  double obj_value;
  double obj_value_prev;
  double norm_value;
  double loss_value;
  double zero_one_error;
  double test_loss;
  double test_error;
  long converge_iter;
  double epsilonVal;
  int convergence_num = 0;
  int converged = 0;
  double EPSILON_VAL = 0.0001;
  // get lambda, max_iter, exam_per_iter from learning parameter
  jclass lpCls = env->GetObjectClass(lp);
  jfieldID lambdaID = env->GetFieldID(lpCls, "lambda", "D");
  jfieldID max_iterID = env->GetFieldID(lpCls, "max_iter", "I");
  jfieldID exam_per_iterID = env->GetFieldID(lpCls, "exam_per_iter", "I");
  jfieldID replaceID = env->GetFieldID(lpCls, "replace", "I");
  jfieldID iterID = env->GetFieldID(lpCls, "iter", "I");
  double lambda = env->GetDoubleField(lp, lambdaID);
  int max_iter = env->GetIntField(lp, max_iterID);
  int exam_per_iter = env->GetIntField(lp, exam_per_iterID);
  int replace = env->GetIntField(lp, replaceID);
  int iter = env->GetIntField(lp, iterID);
  /**
   * @function LearnReturnLast  Native learning method called by JNI
   * //Input parameters
   * @param Dataset std::vector<simple_sparse_vector>& training dataset
   * @param Labels std::vector<int>& Labels of training dataset
   * @param dimension unit dimension of training input
   * @param testDataset std::vector<simple_sparse_vector>& test dataset
   * @param testLabels std::vector<int>& Lables of test dataset
   * @param lambda double regularization parameter
   * @param max_iter int maximum iteration
   * @param exam_per_iter int number of examples per iteration to consider \
   *        for stochastic gradient
   * @param  model_filename std::string& filename to which model will be written
   * //Output parameters
   * @param train_time long& time spend in training
   * @param calc_obj_time long& time spend in objective function calculation
   * @param obj_value double& objective function value
   * @param norm_value double& norm of weight after training
   * @param loss_value double& loss value after training
   * @param zero_one_error double& zero-one misclassification error
   * //Addtional parameters
   * @param eta_rule_type int rule to calculate eta_constant
   * @param eta_constant double learning rate
   * @param projection_rule rule to calculate projection, like cosine, etc
   * @param projection_constant double projection constant
   * @param replace bool replace parameter, set true to reset the weights, false to load from model file
   */
  LearnReturnLast(Dataset, Labels, dimension, testDataset, testLabels,
		  lambda, max_iter, exam_per_iter, replace, iter, modelFile, trainTime,
		  calc_obj_time, obj_value, obj_value_prev, norm_value, loss_value, zero_one_error, 
		  test_loss, test_error, converge_iter, epsilonVal, 0, 0.0, 0, 0.0);

  // -------------------------------------------------------------
  // ---------------------- Print Results ------------------------
  // -------------------------------------------------------------
  std::ofstream fs;

  // create a name for the file output
  std::string filename = modelFile + "_results.csv";















if(replace == 0){

vector<float> values = GetLastLine(filename);
convergence_num = static_cast<int>(values.end()[-2]);
converged = static_cast<int>(values.end()[-1]);
//std::cout << "Convergence_num: " << convergence_num << " Converged: " << converged;

// Algorithm to check for convergence

/*
Check for obj_value_prev and obj_value, if 
*/


if(epsilonVal <= EPSILON_VAL && converged==0){
  //cout << "\nEpsilon Val: " << epsilonVal << " .Incrementing convergence_num by 1.";
  convergence_num++;
  if(convergence_num == 10){
    converged=1;
    //cout << "Algorithm converged on this node at iter " << iter;
  }
  
}

if(epsilonVal > EPSILON_VAL && converged==0){
// reset convergence_num to zero.
  //cout << "\nEpsilon Val: " << epsilonVal << " .Resetting convergence_num to zero.";
  convergence_num = 0;

}


/*
for (std::vector<float>::const_iterator i = values.begin(); i != values.end(); ++i)
    std::cout << *i << ' ';
*/

  fs.open(filename.c_str(), std::ios::out | std::ios::app);
fs << "\n" << iter << "," << obj_value << "," << zero_one_error << "," << loss_value << "," << convergence_num << "," << converged;

}

else {

  fs.open(filename.c_str(), std::ios::out);

  fs << "Iter,PrimalObjective,ZeroOneError,LossValue,ConvergenceNum,Converged\n";
  fs << iter<< "," << obj_value << "," << zero_one_error << "," << loss_value << "," << convergence_num << ","<< converged;
}

  /*
  std::cout << readingTime << " = Reading time\n"
      << trainTime << " = Model training time\n"
      << calc_obj_time << " = Time to calculate the objective\n"
      << epsilonVal << " = Epsilon at convergence\n" 
      << converge_iter << " = Convergence iteration\n"
      << norm_value  << " = Norm of solution\n" 
	    << loss_value << " = avg Loss of solution\n"  
	    << zero_one_error  << " = avg zero-one error of solution\n" 
	    << obj_value << " = primal objective of solution\n" 
	    <<  std::endl;
 
   
              */
 //std::cout <<  std::endl 
 //             << "Objective Value before training: " << obj_value_prev << std::endl
 //             << "Objective Value after training: " << obj_value << std::endl;
              
/*
    fs << readingTime << " = Reading time " << ','
      << trainTime << " = Model training time " << ','
      << calc_obj_time << " = Time to calculate the objective " << ','
      << epsilonVal << " = Epsilon at convergence " << ','
      << converge_iter << " = Convergence iteration " << ','
      << norm_value  << " = Norm of solution " << ',' 
      << loss_value << " = avg Loss of solution " << ',' 


      << zero_one_error  << " = avg zero-one error of solution " << ',' 
      << obj_value << " = primal objective of solution" 
      <<  std::endl;
*/
      fs.close();
}

JNIEXPORT void JNICALL Java_jnipegasos_JNIPegasosInterface_classify
(JNIEnv * env, jobject obj, jstring testfilename, jstring modelfilename, jstring predictfilename) {
  // get filenames from arguments by converting jstring to native string
  const char* ctestfilename = env->GetStringUTFChars(testfilename, 0);
  const char* cmodelfilename = env->GetStringUTFChars(modelfilename, 0);
  const char* cpredictfilename = env->GetStringUTFChars(predictfilename, 0);
  std::string testFile(ctestfilename);
  std::string s_modelFile(cmodelfilename);
  std::string s_predictFile(cpredictfilename);
  env->ReleaseStringUTFChars(testfilename, ctestfilename);
  env->ReleaseStringUTFChars(modelfilename, cmodelfilename);
  env->ReleaseStringUTFChars(predictfilename, cpredictfilename);

  // read the testfile and initialize dataset and label vector
  uint dimension = 0; // dimension of test and train data should be same???
  std::vector<simple_sparse_vector> TestDataset;
  std::vector<int> TestLabels;
  double readingTime;
  ReadData(testFile, TestDataset, TestLabels, dimension, readingTime);

  uint num_examples = TestLabels.size();
  
  // Initialization of classification vector
  // read modelfile and initialize weightvector
  std::ifstream modelFile(s_modelFile.c_str(), std::ifstream::in);
  if (!modelFile.good()) {
    std::cerr << "error w/ " << s_modelFile << std::endl;
    exit(EXIT_FAILURE);
  }
  std::vector<int> PredictedLabels(num_examples);
  WeightVector W(dimension, modelFile);
  double loss_value = 0.0;
  int misclassified_count = 0;
  // Now, in a loop predict, fill the predict vector and also write in predictfile,
  // also do misclassified count, loss, zero-one-loss, etc..
  for (uint i=0; i < num_examples; ++i) {
    double predict = W * TestDataset[i];
    double cur_loss = 1 - TestLabels[i] * predict; 
    if (cur_loss < 0.0) cur_loss = 0.0;
    loss_value += cur_loss/num_examples;
    if (cur_loss >= 1.0) {
      misclassified_count++;
      PredictedLabels[i] = -1 * TestLabels[i];
    }
    else
      PredictedLabels[i] = TestLabels[i];
  }
  
  // write predicted labels to predict file
  std::ofstream predictFile(s_predictFile.c_str());
  if (!predictFile.good()) {
    std::cerr << "error w/ " << s_predictFile << std::endl;
    exit(EXIT_FAILURE);
  }
  for (uint i=0; i < num_examples; ++i) {
    predictFile << PredictedLabels[i] << std::endl;
  }
  std::cout << s_predictFile << " = predicted labels file" << std::endl;
  predictFile.close();
  
  // -------------------------------------------------------------
  // ---------------------- Print Results ------------------------
  // -------------------------------------------------------------
  std::cout << loss_value << " = avg Loss of solution\n"  
	    << "(" << misclassified_count <<"/" << num_examples << ")" << " = number of misclassified examples\n" 
	    <<  std::endl;
}





