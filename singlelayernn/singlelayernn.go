package singlelayernn

import "naiTasks/perceptron"

type SingleLayerNeuralNetwork struct {
	neurons []perceptron.Perceptron

	// learning rate
	alpha float64

	// bias/threshold
	beta float64
}

func NewSingleLayerNeuralNetwork(perceptrons []perceptron.Perceptron, alpha, beta float64) *SingleLayerNeuralNetwork {
	return &SingleLayerNeuralNetwork{
		neurons: perceptrons,
		alpha:   alpha,
		beta:    beta,
	}
}

func (slnn *SingleLayerNeuralNetwork) TrainLayer(inputs [][]float64, labels []int) error {
	// todo
}
