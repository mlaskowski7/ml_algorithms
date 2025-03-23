package perceptron

import "testing"

func TestNewPerceptron(t *testing.T) {
	dimension := 2
	weights := []float64{0.5, -0.5}
	threshold := 0.2
	learningRate := 0.1

	p, err := NewPerceptron(dimension, weights, threshold, learningRate)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if p.dimension != dimension {
		t.Errorf("Expected dimension %d, got %d", dimension, p.dimension)
	}

	if len(p.weights) != dimension {
		t.Errorf("Expected weights length %d, got %d", dimension, len(p.weights))
	}

	if p.threshold != threshold {
		t.Errorf("Expected threshold %.2f, got %.2f", threshold, p.threshold)
	}

	if p.learningRate != learningRate {
		t.Errorf("Expected learningRate %.2f, got %.2f", learningRate, p.learningRate)
	}
}

func TestPredict(t *testing.T) {
	p, _ := NewPerceptron(2, []float64{1.0, 1.0}, 1.5, 0.1)

	// Should predict 1 (2*1.0 - 1.5 >= 0)
	input := []float64{1.0, 1.0}
	prediction, err := p.Predict(input)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}
	if prediction != 1 {
		t.Errorf("Expected prediction 1, got %d", prediction)
	}

	// Should predict 0 (0.5*1.0 - 1.5 < 0)
	input = []float64{1.0, 0.0}
	prediction, err = p.Predict(input)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}
	if prediction != 0 {
		t.Errorf("Expected prediction 0, got %d", prediction)
	}
}

func TestPredictDimensionMismatch(t *testing.T) {
	p, _ := NewPerceptron(2, []float64{0.0, 0.0}, 0.0, 0.1)

	// Pass input of wrong length
	_, err := p.Predict([]float64{1.0, 2.0, 3.0})
	if err == nil {
		t.Error("Expected error due to dimension mismatch, got nil")
	}
}

func TestTrainANDGate(t *testing.T) {
	// AND gate dataset
	inputs := [][]float64{
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 1.0},
	}
	labels := []int{0, 0, 0, 1}

	// Initialize perceptron with zero weights
	p, err := NewPerceptron(2, []float64{0.0, 0.0}, 0.0, 0.1)
	if err != nil {
		t.Fatalf("NewPerceptron returned error: %v", err)
	}

	epochs, err := p.Train(inputs, labels)
	if err != nil {
		t.Fatalf("Train returned error: %v", err)
	}

	// Test predictions after training
	for i, input := range inputs {
		prediction, err := p.Predict(input)
		if err != nil {
			t.Errorf("Predict returned error on input %v: %v", input, err)
		}
		if prediction != labels[i] {
			t.Errorf("Expected prediction %d for input %v, got %d", labels[i], input, prediction)
		}
	}

	t.Logf("Training completed in %d epochs", epochs)
}
