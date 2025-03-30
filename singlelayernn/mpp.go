package singlelayernn

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"naiTasks/commons"
	"os"
	"strconv"
	"strings"
)

func TransformStringToSliceWithLettersCount(text string) []float64 {
	result := make([]float64, 26)
	text = strings.ToLower(text)
	letters := 0

	for _, c := range text {
		if c >= 'a' && c <= 'z' {
			index := c - 'a'
			result[index]++
			letters++
		}
	}

	for i := 0; i < 26; i++ {
		result[i] /= float64(letters)
	}

	return result
}

func LoadLanguagesDatasetFromCsv(path string) ([][]float64, []int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open CSV file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read CSV file: %w", err)
	}

	if len(records) < 2 {
		return nil, nil, fmt.Errorf("CSV file is empty or missing data")
	}

	var inputs [][]float64
	var labels []int

	for i, row := range records {
		if i == 0 {
			continue
		}
		if len(row) < 2 {
			return nil, nil, fmt.Errorf("invalid row at line %d", i+1)
		}

		text := row[0]
		labelStr := row[1]

		sentence := TransformStringToSliceWithLettersCount(text)
		label, err := strconv.Atoi(labelStr)
		if err != nil {
			return nil, nil, fmt.Errorf("invalid label at line %d: %w", i+1, err)
		}

		inputs = append(inputs, sentence)
		labels = append(labels, label)
	}

	return inputs, labels, nil
}

func TestLanguageNeuralNetwork() (string, error) {

	// classes:
	// 0 - polish
	// 1 - english
	// 3 - german

	languageNeuralNetwork, err := NewSingleLayerNeuralNetwork(3, 26, 0.1, 0.2)
	if err != nil {
		return "", err
	}

	inputs, labels, err := LoadLanguagesDatasetFromCsv("data/languages.csv")
	if err != nil {
		return "", err
	}

	trainInputs, trainLabels, testInputs, testLabels := commons.TrainTestSplitWithLabels(inputs, labels)

	err = languageNeuralNetwork.TrainLayer(trainInputs, trainLabels)
	if err != nil {
		return "", err
	}

	predictedLabels := make([]int, len(testLabels))
	for i, input := range testInputs {
		predicted, err := languageNeuralNetwork.Predict(input)
		if err != nil {
			fmt.Printf("prediction failed on sample %d: %v\n", i, err)
			continue
		}

		predictedLabels[i] = predicted
	}

	accuracy := commons.MeasureAccuracy(testLabels, predictedLabels)
	fmt.Printf("Accuracy is: %d%%", accuracy)

	numClasses := 3
	for class := 0; class < numClasses; class++ {
		tp, fp, fn := 0, 0, 0
		for i := range testLabels {
			actual := testLabels[i]
			predicted := predictedLabels[i]

			if predicted == class && actual == class {
				tp++
			} else if predicted == class && actual != class {
				fp++
			} else if predicted != class && actual == class {
				fn++
			}
		}

		precision := commons.Precision(tp, fp)
		recall := commons.Recall(tp, fn, fp)
		f1 := commons.Fmeasure(int(precision*100), int(recall*100)) / 100

		fmt.Printf("\nClass %d metrics:\n", class)
		fmt.Printf("  - Precision: %.2f\n", precision)
		fmt.Printf("  - Recall:    %.2f\n", recall)
		fmt.Printf("  - F1-score:  %.2f\n", f1)
	}

	err = predictWithUi(languageNeuralNetwork)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%d%%", accuracy), nil
}

func predictWithUi(languageNeuralNetwork *SingleLayerNeuralNetwork) error {
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Enter a sentence to classify (Polish or English or German): ")
	inputText, _ := reader.ReadString('\n')

	vector := TransformStringToSliceWithLettersCount(inputText)
	predicted, err := languageNeuralNetwork.Predict(vector)
	if err != nil {
		return err
	}

	languageMap := map[int]string{
		0: "Polish",
		1: "English",
		2: "German",
	}

	fmt.Printf("Predicted language is %s (class %d)\n", languageMap[predicted], predicted)
	return nil
}
