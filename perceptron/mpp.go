package perceptron

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math/rand"
	"naiTasks/commons"
	"os"
	"strconv"
	"strings"
)

func TrainPerceptronOnIrisCsv(filePath string) error {
	inputs, labels, err := getInputsAndLabelsFromCsv(filePath)
	if err != nil {
		return err
	}

	trainInputs, trainLabels, testInputs, testLabels := commons.TrainTestSplitWithLabels(inputs, labels)

	// random weights at start
	weights := make([]float64, len(trainInputs[0]))
	for i := range weights {
		weights[i] = rand.Float64()*0.1 - 0.05
	}

	perceptron, err := NewPerceptron(len(weights), weights, 0.0, 0.1)
	if err != nil {
		return fmt.Errorf("failed to initialize perceptron perceptron: %v", err)
	}

	epochs, err := perceptron.Train(trainInputs, trainLabels)
	if err != nil {
		return fmt.Errorf("error occured while training the perceptron: %v", err)
	}
	fmt.Printf("Training finished in %d epochs\n", epochs)

	predictedLabels := make([]int, len(testLabels))
	for index, _ := range testInputs {
		predictedLabel, err := perceptron.Predict(testInputs[index])
		if err != nil {
			return fmt.Errorf("error occured while predicting the test label: %v", err)
		}
		predictedLabels[index] = predictedLabel
	}

	testAccuracy := commons.MeasureAccuracy(testLabels, predictedLabels)
	fmt.Printf("Final perceptron accuracy after testing is %v%%", testAccuracy)
	listenForObservationInput(perceptron)
	return nil
}

func listenForObservationInput(perceptron *Perceptron) {
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Println("\nEnter sepal length, sepal width, petal length, petal width separated by commas to start prediction:")

		inputLine, err := reader.ReadString('\n')
		if err != nil {
			fmt.Printf("Failed to read input: %v\n", err)
			continue
		}

		inputLine = strings.TrimSpace(inputLine)

		parts := strings.Split(inputLine, ",")
		if len(parts) != 4 {
			fmt.Println("Please enter exactly 4 floats separated by space.")
			continue
		}

		observation := make([]float64, 4)
		for i, val := range parts {
			num, err := strconv.ParseFloat(val, 64)
			if err != nil {
				fmt.Printf("Cant parse '%s'", val)
				continue
			}
			observation[i] = num
		}

		prediction, err := perceptron.Predict(observation)
		if err != nil {
			fmt.Printf("Cant predict: %v\n", err)
			continue
		}

		className := ""
		if prediction == 1 {
			className = "Iris-setosa"
		} else {
			className = "Iris-versicolor"
		}

		fmt.Printf("Result: %d (%s)\n", prediction, className)
	}
}

func getInputsAndLabelsFromCsv(filePath string) ([][]float64, []int, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open csv file: %v", err)
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			fmt.Printf("failed to close csv file: %v", err)
		}
	}(file)

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read csv file: %v", err)
	}

	var inputs [][]float64
	var labels []int
	for _, row := range rows[1:] {
		features := make([]float64, 4)
		for i := 0; i < 4; i++ {
			features[i], _ = strconv.ParseFloat(row[i], 64)
		}

		var label int
		if row[4] == "Iris-setosa" {
			label = 1
		} else if row[4] == "Iris-versicolor" {
			label = 0
		} else {
			continue
		}

		inputs = append(inputs, features)
		labels = append(labels, label)
	}

	return inputs, labels, nil
}
