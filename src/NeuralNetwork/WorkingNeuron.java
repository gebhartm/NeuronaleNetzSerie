package NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

import NeurakNetwork.ActivationFunctions.ActivationFunction;
import NeurakNetwork.ActivationFunctions.Boolean;
import NeurakNetwork.ActivationFunctions.Identity;

//Hidden und Output Neuron !!!

public class WorkingNeuron extends Neuron {
	private List<Connection> connections = new ArrayList<>();
	private ActivationFunction activationFunction = ActivationFunction.ActivationSigmoid;
	
	@Override
	public float getValue() {
		float sum = 0;
		for(Connection c : connections) {
			sum += c.getValue();
		}
		
		//return sum;
		return activationFunction.activation(sum);
	}
	
	public void addConnection(Connection c) {
		connections.add(c);
	}
	
	public void deltaLearning(float epsilon, float smallDelta) {
		for(int i = 0; i < connections.size(); i++) {
			float bitDelta = epsilon * smallDelta * connections.get(i).getValue();
			connections.get(i).addWeight(bitDelta);
		}
	}

}
