#ifndef NET_H
#define NET_H

#include <iostream>
#include <vector>
#include <cassert>
#include "Unit.h"
using namespace std;
class Net
{
public:

	/*
	    You should *not* change this part
	*/

	// constructor. 
	// topology is a container representing net structure. 
	//   e.g. {2, 4, 1} represents 2 neurons for the first layer, 4 for the second layer, 1 for last layer
	// if you want to hard-code the structure, just ignore the variable topology 
	// eta: learning rate 
	Net(const std::vector<unsigned> &topology, const double eta);

	Net(const char inputFile[], const double eta);

	// given an input sample inputVals, propagate input forward, compute the output of each neuron 
	void feedForward(const std::vector<double> &inputVals);

	// given the vector targetVals (ground truth of output), propagate errors backward, and update each weights
	void backProp(const std::vector<double> &targetVals);

	// output the prediction for the current sample to the vector resultVals
	void getResults(std::vector<double> &resultVals) const;

	// return the error of the current sample
	double getError(void) const;

	void saveStateToFile(const char[]) ;

	
	/*
	    Add what you need in the below
	*/

	
	// ...

private:
	// ...
	vector<vector<Unit> > network;
	vector<vector<double> > currentOutput;
	double learningRate;
	vector<double> target;
};




#endif//NET_H
