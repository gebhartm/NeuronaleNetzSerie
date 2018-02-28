package NeuralNetwork;

public class Connection {
	private Neuron neuron;
	private float weight;
	
	public Connection(Neuron neuron, float weight) {
		this.neuron = neuron;
		this.weight = weight;
	}
	
	public float getValue() {
		return neuron.getValue() * weight;
	}
	
	public void addWeight(float weightDelta) {
		weight += weightDelta;
	}
}
