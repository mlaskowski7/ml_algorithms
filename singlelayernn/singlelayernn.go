package singlelayernn

import "naiTasks/perceptron"

type SingleLayerNeuralNetwork struct {
	neurons []*perceptron.Perceptron

	// learning rate
	alpha float64

	// bias/threshold
	beta float64
}

func NewSingleLayerNeuralNetwork(classesLen, neuronDimension int, alpha, beta float64) (*SingleLayerNeuralNetwork, error) {
	neurons := make([]*perceptron.Perceptron, classesLen)
	for i := 0; i < classesLen; i++ {
		initialWeights := make([]float64, neuronDimension)
		for i := range initialWeights {
			initialWeights[i] = 0
		}
		var err error
		neurons[i], err = perceptron.NewPerceptron(neuronDimension, initialWeights, beta, alpha)
		if err != nil {
			return nil, err
		}
	}
	return &SingleLayerNeuralNetwork{
		neurons: neurons,
		alpha:   alpha,
		beta:    beta,
	}, nil
}

func (slnn *SingleLayerNeuralNetwork) TrainLayer(inputs [][]float64, labels []int) error {
	for classIndex, neuron := range slnn.neurons {
		trainLabels := make([]int, len(labels))
		for labelIndex, label := range labels {
			if label == classIndex {
				trainLabels[labelIndex] = 1
			} else {
				trainLabels[labelIndex] = 0
			}
		}

		_, err := neuron.Train(inputs, trainLabels)
		if err != nil {
			return err
		}
	}

	return nil
}

func (slnn *SingleLayerNeuralNetwork) Predict(inputs []float64) (int, error) {
	var maxNetValue float64
	var predictionResult int
	for class, neuron := range slnn.neurons {
		netValue := 0.0

		for i := 0; i < len(inputs); i++ {
			netValue += neuron.Weights()[i] * inputs[i]
		}
		netValue -= neuron.Threshold()

		if netValue > maxNetValue {
			maxNetValue = netValue
			predictionResult = class
		}
	}

	return predictionResult, nil
}
