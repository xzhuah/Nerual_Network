#include "TrainingData.h"
#include "Net.h"



#define ETA 0.5 // net learning rate, [0.0..1.0]


void showVectorVals(std::string label, std::vector<double> &v)
{
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }

    std::cout << std::endl;
}
void train(char inputfile[]) {
	TrainingData trainData(inputfile);
	// e.g., { 2, 4, 1 }
	std::vector<unsigned> topology;

	// get the structure of the net
	trainData.getTopology(topology);

	Net myNet(topology, ETA);

	std::vector<double> inputVals, targetVals, resultVals;
	int trainingIteration = 0;
	///////////////////
	int mycontrol = 0;
	while ((!trainData.isEof()) && (mycontrol<10)) {
		mycontrol++;
		++trainingIteration;
		std::cout << std::endl << "Iteration " << trainingIteration;

		// Get new input data and feed it forward:
		if (trainData.getNextInputs(inputVals) != topology[0]) {
			break;
		}
		showVectorVals(": Inputs:", inputVals);

		// feed Forward	
		myNet.feedForward(inputVals);

		// Collect the net's actual output results:
		myNet.getResults(resultVals);
		showVectorVals("Outputs:", resultVals);

		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		// Report how well the training is working, average over recent samples:
		std::cout << "loss: "
			<< myNet.getError() << std::endl;
	}
	std::cout << std::endl << "Training Complete" << std::endl;
	myNet.saveStateToFile("./trainResult.txt");
}
void continueTrain(char inputfile[], char networkfile[]) {
	TrainingData trainData(inputfile);
	// e.g., { 2, 4, 1 }
	std::vector<unsigned> topology;

	// get the structure of the net
	trainData.getTopology(topology);

	Net myNet(networkfile, ETA);
	

	std::vector<double> inputVals, targetVals, resultVals;
	int trainingIteration = 0;
	
	
	while (!trainData.isEof()) {
		
		++trainingIteration;
		std::cout << std::endl << "Iteration " << trainingIteration;

		// Get new input data and feed it forward:
		if (trainData.getNextInputs(inputVals) != topology[0]) {
			break;
		}
		showVectorVals(": Inputs:", inputVals);

		// feed Forward	
		myNet.feedForward(inputVals);

		// Collect the net's actual output results:
		myNet.getResults(resultVals);
		showVectorVals("Outputs:", resultVals);

		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		// Report how well the training is working, average over recent samples:
		std::cout << "loss: "
			<< myNet.getError() << std::endl;
	}
	std::cout << std::endl << "Training Complete" << std::endl;
	myNet.saveStateToFile("./trainResult.txt");
}
int main()
{
	
	continueTrain("./trainingData.txt","./trainResult.txt");
	//train("./trainingData.txt");
	
	int tt;//Avoid stop 
	std::cin >> tt;
}
