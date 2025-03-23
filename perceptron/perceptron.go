package perceptron

import (
	"encoding/csv"
	"fmt"
	"naiTasks/commons"
	"os"
	"strconv"
	"time"
)

type Perceptron struct {
	dimension    int
	weights      []float64
	threshold    float64
	learningRate float64
}

func NewPerceptron(dimension int, weights []float64, threshold float64, learningRate float64) (*Perceptron, error) {
	if dimension != len(weights) {
		return nil, fmt.Errorf("dimensions(%d) do not match the weights length(%d)", dimension, len(weights))
	}

	return &Perceptron{
		dimension,
		weights,
		threshold,
		learningRate,
	}, nil
}

func (p *Perceptron) Predict(inputs []float64) (int, error) {
	if len(inputs) != p.dimension {
		return 0, fmt.Errorf("inputs length(%d) do not match the dimension(%d)", len(inputs), p.dimension)
	}
	var sum float64
	for index, input := range inputs {
		sum += input * p.weights[index]
	}

	if sum-p.threshold >= 0 {
		return 1, nil
	} else {
		return 0, nil
	}
}

func (p *Perceptron) Train(inputs [][]float64, labels []int) (int, error) {
	var epochCounter int

	timestamp := time.Now().Format("20060102_150405")
	fileName := fmt.Sprintf("perceptron_accuracy_%s.csv", timestamp)
	file, err := os.Create("data/" + fileName)
	if err != nil {
		return 0, fmt.Errorf("failed to create csv file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	err = writer.Write([]string{"Epoch", "Errors", "Accuracy"})
	if err != nil {
		return 0, fmt.Errorf("failed to write to csv file: %v", err)
	}

	for {
		errors := 0
		predictedLabels := make([]int, len(labels))
		for index, input := range inputs {
			expectedPrediction := labels[index]
			prediction, err := p.Predict(input)
			if err != nil {
				return 0, err
			}

			predictedLabels[index] = prediction
			if prediction != expectedPrediction {
				errors += 1
			}

			// apply delta rule to weights and threshold
			for j := 0; j < p.dimension; j++ {
				p.weights[j] += float64(expectedPrediction-prediction) * p.learningRate * input[j]
			}
			p.threshold -= float64(expectedPrediction-prediction) * p.learningRate
		}
		accuracy := commons.MeasureAccuracy(labels, predictedLabels)
		epochCounter += 1
		fmt.Printf("Accuracy in epoch no.%d: %d%%\n", epochCounter, accuracy)
		fmt.Printf("Epoch no.%d had %d errors\n", epochCounter, errors)

		err := writer.Write([]string{
			strconv.Itoa(epochCounter),
			fmt.Sprintf("%d", errors),
			fmt.Sprintf("%d", accuracy),
		})
		if err != nil {
			return 0, fmt.Errorf("failed to write to csv file: %v", err)
		}

		if errors <= 0 {
			break
		}
	}

	return epochCounter, nil
}
