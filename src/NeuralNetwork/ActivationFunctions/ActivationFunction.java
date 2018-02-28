package NeuralNetwork.ActivationFunctions;

public interface ActivationFunction {
	public static Boolean ActivationBoolean = new Boolean();
	public static Identity ActivationIdentity = new Identity();
	public static Sigmoid ActivationSigmoid = new Sigmoid();
	public static HyperbolicTangent ActivationHyperbolicTangent = new HyperbolicTangent(); 
	
	public float activation(float input);
}
