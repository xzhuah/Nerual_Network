/*
 * Unit.h
 *
 *  Created on: Apr 24, 2016
 *      Author: Xinyu
 */

#ifndef UNIT_H_
#define UNIT_H_

#include <vector>
#include <cmath>
#include <stdio.h>     
#include <stdlib.h>
#include <time.h>
using namespace std;

class Unit{

public:

	Unit(unsigned num) {
		
		this->weights.resize(num);//set to random small number
		for (unsigned i = 0; i < num; i++) {
			//this->weights[i] = ((double)rand() / RAND_MAX)*((double)rand() / RAND_MAX)*10.0;
			//this->weights[i] =(rand()%10000-500)/10000.0;
			this->weights[i] = (((double)rand() / RAND_MAX) -0.5)*10;
		}
	}

	Unit(vector<double> weight) {

		this->weights.resize(weight.size());//set to random small number
		for (unsigned i = 0; i < weight.size(); i++) {
			//this->weights[i] = ((double)rand() / RAND_MAX)*((double)rand() / RAND_MAX)*10.0;
			//this->weights[i] =(rand()%10000-500)/10000.0;
			this->weights[i] = weight[i];
		}
	}
	


	double getOutput(const vector<double>& input) const {
		double result = 0;
		for (unsigned i = 0; i < this->weights.size(); i++) {
			result += this->weights[i] * input[i];
		}
		return this->activation(result);
	}

	//For input unit, output directly
	double getOutput(double input) const {
		return input;
	}
	void setWeights(const vector<double>& newWeights) {
		this->weights = newWeights;
	}

	vector<double>* getWeights() {
		return &(this->weights);
	}

private:
	vector<double> weights;

	

	double activation(double weightedSum) const {
		return 1.0 / (1.0 + exp(-weightedSum));
	}
};



#endif /* UNIT_H_ */
