package commons

import (
	"encoding/csv"
	"io"
	"math"
	"math/rand"
	"naiTasks/structs"
	"os"
)

const trainSetPercent = 0.70

func TrainTestSplit(dataset *[]structs.Vector) ([]structs.Vector, []structs.Vector) {
	dereferencedDataset := *dataset

	classGroups := make(map[string][]structs.Vector)
	for _, sample := range dereferencedDataset {
		classGroups[sample.Class()] = append(classGroups[sample.Class()], sample)
	}

	var trainSet []structs.Vector
	var testSet []structs.Vector

	for _, group := range classGroups {
		rand.Shuffle(len(group), func(i, j int) {
			group[i], group[j] = group[j], group[i]
		})

		trainLen := int(math.Round(trainSetPercent * float64(len(group))))

		trainSet = append(trainSet, group[:trainLen]...)
		testSet = append(testSet, group[trainLen:]...)
	}

	return trainSet, testSet
}

func TrainTestSplitWithLabels(inputs [][]float64, labels []int) (trainInputs [][]float64, trainLabels []int, testInputs [][]float64, testLabels []int) {
	rand.Shuffle(len(inputs), func(i, j int) {
		inputs[i], inputs[j] = inputs[j], inputs[i]
		labels[i], labels[j] = labels[j], labels[i]
	})

	splitIndex := int(float64(len(inputs)) * trainSetPercent)
	trainInputs, trainLabels = inputs[:splitIndex], labels[:splitIndex]
	testInputs, testLabels = inputs[splitIndex:], labels[splitIndex:]

	return trainInputs, trainLabels, testInputs, testLabels
}

func ClassificationTrainTestSplit(dataset []map[string]string, testRows int) (trainSet []map[string]string, testSet []map[string]string) {
	rand.Shuffle(len(dataset), func(i, j int) {
		dataset[i], dataset[j] = dataset[j], dataset[i]
	})

	testSet = dataset[:testRows]
	trainSet = dataset[testRows:]
	return trainSet, testSet
}

func LoadDatasetFromCsv(filePath string) ([]map[string]string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}

	defer file.Close()

	csvReader := csv.NewReader(file)

	headersRow, err := csvReader.Read()
	if err != nil {
		return nil, err
	}

	dataset := make([]map[string]string, 0)

	for {
		row, err := csvReader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		data := make(map[string]string)
		for i, header := range headersRow {
			data[header] = row[i]
		}
		dataset = append(dataset, data)
	}

	return dataset, nil
}
