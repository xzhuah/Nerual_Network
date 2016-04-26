#include "Net.h"
#include "Unit.h"
#include <vector>
#include <stdio.h>     
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <sstream>
// constructor.
	// topology is a container representing net structure.
	//   e.g. {2, 4, 1} represents 2 neurons for the first layer, 4 for the second layer, 1 for last layer
	// if you want to hard-code the structure, just ignore the variable topology
	// eta: learning rate
using namespace std;
Net::Net(const std::vector<unsigned> &topology, const double eta){
	srand(time(NULL));
	learningRate = eta;
	for (int i = 0; i < topology.size(); i++) {
		
		unsigned weightNum = (i == 0 ? 1 : topology[i - 1]);
		
		vector<Unit> temp;
		for (int j = 0; j < topology[i]; j++) {
			Unit u(weightNum);
			temp.push_back(u);
		}
		this->network.push_back(temp);
	}
	currentOutput.resize(this->network.size());
	for (int i = 0; i < currentOutput.size(); i++) {
		currentOutput[i].resize(this->network[i].size());
	}

	
}
Net::Net(const char inputFile[], const double eta) {
	learningRate = eta;
	std::ifstream networkInput;
	std::vector<unsigned> Unitnum;
	networkInput.open(inputFile);
	std::string line;
	std::string label;
	getline(networkInput, line);
	std::stringstream ss(line);
	
	unsigned Unitnumber = 0;
	// First line define the structure of the network 2 3 4 (two input three hiddden four output)
	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		Unitnumber += n;
		Unitnum.push_back(n);
	}

	for (int i = 0; i < Unitnum.size(); i++) {
		vector<Unit> layer;
		for (int j = 0; j < Unitnum[i]; j++) {

			getline(networkInput, line);
			
			if (line == "") break;
			std::stringstream ss(line);
			vector<double> weightReader;
			double oneValue;
			while (ss >> oneValue) {
				weightReader.push_back(oneValue);
			}
			Unit u(weightReader);
			layer.push_back(u);
		}
	
		if(layer.size()!=0) this->network.push_back(layer);
		

	}
	networkInput.close();
	currentOutput.resize(this->network.size());
	for (int i = 0; i < currentOutput.size(); i++) {
		currentOutput[i].resize(this->network[i].size());
	}

	
	

}
// given an input sample inputVals, propagate input forward, compute the output of each neuron
void Net::feedForward(const std::vector<double> &inputVals){
	
	for (int i = 0; i < currentOutput[0].size(); i++) {
		currentOutput[0][i] = this->network[0][i].getOutput(inputVals[i]);//for input unit, simply output the result
	}
	for (int i = 1; i < this->network.size(); i++) {
		for (int j = 0; j < currentOutput[i].size(); j++) {
			currentOutput[i][j] = this->network[i][j].getOutput(currentOutput[i - 1]);
		}
	}
}

// given the vector targetVals (ground truth of output), propagate errors backward, and update each weights
void Net::backProp(const std::vector<double> &targetVals){
	this->target = targetVals;
	//Get sigma for output units
	vector<vector<double> > sigma(network.size());
	sigma[sigma.size() - 1].resize(network[sigma.size() - 1].size());
	for (int i = 0; i < sigma[sigma.size() - 1].size(); i++) {
		double oi = currentOutput[sigma.size() - 1][i];
		sigma[sigma.size() - 1][i] = oi*(1 - oi)*(targetVals[i] - oi);

		/*for output information
		cout << "Sigma" << sigma.size() - 1 << "," << i << " is " << sigma[sigma.size() - 1][i];

		cout << " Output" << sigma.size() - 1 << "," << i << " is " << currentOutput[sigma.size() - 1][i];
		cout << " Weight" << sigma.size() - 1 << "," << i << " is ";
		for (int ii = 0; ii < network[sigma.size() - 1][i].getWeights().size(); ii++) {
			cout << network[sigma.size() - 1][i].getWeights()[ii] << " ";
		}
		cout << endl;*/
	}
	//Get sigma for hidden units
	for (int i = network.size() - 2; i >= 1; i--) {
		sigma[i].resize(network[i].size());
		for (int j = 0; j < sigma[i].size(); j++) {
			double oj = currentOutput[i][j];
			double wsum = 0;
			for (int k = 0; k < network[i + 1].size(); k++) {
				vector<double>* tempWeight = network[i + 1][k].getWeights();
				wsum += ((*tempWeight)[j]*sigma[i+1][k]);
			}
			sigma[i][j] = oj*(1 - oj)*wsum;

			/*
			For output infromation
			cout << "Sigma" << i << "," << j << " is " << sigma[i][j];
			cout << " Output" << i << "," << j << " is " << oj ;
			cout << " Weight" << i << "," << j << " is " ;
			for (int ii = 0; ii < network[i][j].getWeights().size(); ii++) {
				cout << network[i][j].getWeights()[ii]<<" ";
			}
			cout << endl;*/
		}
	}

	//Update weight for hidden unit and output unit
	for (int i = 1; i < network.size(); i++) {
		for (int j = 0; j < network[i].size(); j++) {
			vector<double> *weight = network[i][j].getWeights();
			for (int k = 0; k < weight->size(); k++) {
				(*weight)[k] += learningRate * sigma[i][j] * currentOutput[i - 1][k];
			}
			//network[i][j].setWeights(weight);
		}
	}
}


// output the prediction for the current sample to the vector resultVals
void Net::getResults(std::vector<double> &resultVals) const{
	
	resultVals = currentOutput[currentOutput.size() - 1];
}


// return the error of the current sample
double Net::getError(void) const{
	double result = 0;
	for (int i = 0; i < target.size(); i++) {
		double differ= currentOutput[currentOutput.size() - 1][i] - target[i];
		result += 0.5*(differ*differ);
	}
	return result;
}

void Net::saveStateToFile(const char file[]){
	ofstream out(file);
	for (int i = 0; i < this->network.size(); i++) {
		out << (this->network[i].size())<<" ";
	}
	out << endl;
	for (int i = 0; i < this->network.size(); i++) {
		for (int j = 0; j < this->network[i].size(); j++) {
			vector<double>* weights = this->network[i][j].getWeights();
			
			for (int k = 0; k < weights->size(); k++) {
				out << ((*weights)[k]) << " ";
			}
			out << endl;
			
		}
	}
}
