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

func (p *Perceptron) Weights() []float64 {
	return p.weights
}

func (p *Perceptron) Threshold() float64 {
	return p.threshold
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
	file, err := os.Create("results/" + fileName)
	if err != nil {
		return 0, fmt.Errorf("failed to create csv file: %v", err)
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {

		}
	}(file)

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

	dataFile, err := os.Create("results/perceptron_hyperplane_data" + timestamp + ".csv")
	if err != nil {
		return 0, fmt.Errorf("failed to create plot data file: %v", err)
	}
	defer func(dataFile *os.File) {
		err := dataFile.Close()
		if err != nil {

		}
	}(dataFile)

	dataWriter := csv.NewWriter(dataFile)
	defer dataWriter.Flush()

	err = dataWriter.Write([]string{"x", "y", "label"})
	if err != nil {
		return 0, fmt.Errorf("failed to write header to plot data: %v", err)
	}

	for i, input := range inputs {
		err = dataWriter.Write([]string{
			fmt.Sprintf("%f", input[0]),
			fmt.Sprintf("%f", input[1]),
			strconv.Itoa(labels[i]),
		})
		if err != nil {
			return 0, fmt.Errorf("failed to write input to plot data: %v", err)
		}
	}

	err = dataWriter.Write([]string{
		"#weights",
		fmt.Sprintf("%f", p.weights[0]),
		fmt.Sprintf("%f", p.weights[1]),
		fmt.Sprintf("%f", p.threshold),
	})
	if err != nil {
		return 0, fmt.Errorf("failed to write weights to plot data: %v", err)
	}

	return epochCounter, nil
}
