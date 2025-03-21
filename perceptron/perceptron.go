package perceptron

type Perceptron struct {
	inputs    []Input
	threshold float64
}

type Input struct {
	value  float64
	weight float64
}

func NewPerceptron(inputs []Input, threshold float64) *Perceptron {
	return &Perceptron{
		inputs,
		threshold,
	}
}

func (p *Perceptron) Predict(learningRate float64) int {
	var sum float64
	for _, input := range p.inputs {
		sum += input.value * input.weight
	}

	if sum-p.threshold > 0 {
		return 1
	} else {
		return 0
	}
}
