package knn

import (
	"encoding/csv"
	"fmt"
	"naiTasks/commons"
	"naiTasks/structs"
	"os"
	"strconv"
)

type Dataset struct {
	trainDataset []structs.Vector
	testDataset  []structs.Vector
}

func NewDatasetFromCsv(filename string) (*Dataset, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			fmt.Println("could not close csv file: " + err.Error())
			return
		}
	}(file)

	fileReader := csv.NewReader(file)
	lines, err := fileReader.ReadAll()
	if err != nil {
		return nil, err
	}

	var dataset []structs.Vector

	for _, line := range lines {
		var row []float64
		for i := 0; i < len(line)-1; i++ {
			floatValue, _ := strconv.ParseFloat(line[i], 64)
			row = append(row, floatValue)
		}
		class := line[len(line)-1]
		vec, _ := structs.NewVector([][]float64{row}, class)
		dataset = append(dataset, *vec)
	}

	trainDataSet, testDataSet := commons.TrainTestSplit(&dataset)

	return &Dataset{trainDataSet, testDataSet}, nil
}

func (d *Dataset) PredictWithKnn(observation structs.Vector, k int) (class string, err error) {
	knn := NewKnn(k, d.trainDataset)
	class, err = knn.PerformPrediction(&observation)
	return
}

func (d *Dataset) TestAlgorithm(k int) (string, error) {
	expectedClasses := make([]string, len(d.testDataset))
	actualClasses := make([]string, len(d.testDataset))

	for i, vec := range d.testDataset {
		expectedClasses[i] = vec.Class()
		predicted, err := d.PredictWithKnn(vec, k)
		if err != nil {
			fmt.Printf("ERROR OCCURRED IN K = %d, not evaluating this k further", i)
			return "", err
		}
		actualClasses[i] = predicted
	}

	accuracy := commons.MeasureAccuracy(expectedClasses, actualClasses)
	return fmt.Sprintf("%d%%", accuracy), nil
}
