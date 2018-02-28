package NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
	private List<InputNeuron> inputNeurons = new ArrayList<>();
	private List<WorkingNeuron> hiddenNeurons = new ArrayList<>();
	private List<WorkingNeuron> outputNeurons = new ArrayList<>();
	
	
	
	public WorkingNeuron createNewOutput(){
		WorkingNeuron wn = new WorkingNeuron();
		outputNeurons.add(wn);
		return wn;
	}
	
	public void createHiddenNeuron(int amount){
		for(int i = 0; i < amount; i++) {
			hiddenNeurons.add(new WorkingNeuron());
		}
	}
	
	public InputNeuron createNewInput() {
		InputNeuron in = new InputNeuron();
		inputNeurons.add(in);
		return in;
	}
	
	public void createFullMesh(){
		if(hiddenNeurons.size() == 0) {
			for(WorkingNeuron wn : outputNeurons){
				for(InputNeuron in : inputNeurons){
					wn.addConnection(new Connection(in, 0));
				}
			}
		}else{
			for(WorkingNeuron wn : outputNeurons){
				for(WorkingNeuron hidden : hiddenNeurons){
					wn.addConnection(new Connection(hidden, 0));
				}
			}
			
			for(WorkingNeuron hidden : hiddenNeurons){
				for(InputNeuron in : inputNeurons){
					hidden.addConnection(new Connection(in, 0));
				}
			}
		}
	}
	
	public void deltaLearning(float[] shoulds, float epsilon) {
		if(shoulds.length != outputNeurons.size()){
			throw new IllegalArgumentException();
		}
		
		if(hiddenNeurons.size() != 0) {
			throw new IllegalStateException();
		}
		
		for(int i = 0; i<shoulds.length; i++) {
			float smallDelta = shoulds[i] - outputNeurons.get(i).getValue();
			outputNeurons.get(i).deltaLearning(epsilon, smallDelta);
		}
	}
	
	public void createFullMesh(float...weights){
		if(hiddenNeurons.size() == 0) {
			if(weights.length != inputNeurons.size() * outputNeurons.size()) {
				throw new RuntimeException();
			}
			
			int index = 0;
			
			for(WorkingNeuron wn : outputNeurons){
				for(InputNeuron in : inputNeurons){
					wn.addConnection(new Connection(in, weights[index++]));
				}
			}
		}else{
			if(weights.length != inputNeurons.size() * hiddenNeurons.size() + hiddenNeurons.size() * outputNeurons.size() ) {
				throw new RuntimeException();
			}
			
			int index = 0;
			
			for(WorkingNeuron hidden : hiddenNeurons){
				for(InputNeuron in : inputNeurons){
					hidden.addConnection(new Connection(in,  weights[index++]));
				}
			}
			
			for(WorkingNeuron wn : outputNeurons){
				for(WorkingNeuron hidden : hiddenNeurons){
					wn.addConnection(new Connection(hidden,  weights[index++]));
				}
			}
		}
	}
}
