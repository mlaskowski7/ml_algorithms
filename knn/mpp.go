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
	file, err := os.Open("data/" + filename)
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
		var row []int32
		for i := 0; i < len(line)-1; i++ {
			intValue, _ := strconv.ParseInt(line[i], 0, 64)
			row = append(row, int32(intValue))
		}
		class := line[len(line)-1]
		vec, _ := structs.NewVector([][]int32{row}, class)
		dataset = append(dataset, *vec)
	}

	trainDataSet, testDataSet := commons.TrainTestSplit(&dataset)

	return &Dataset{trainDataSet, testDataSet}, nil
}

func (d *Dataset) PredictWithKnn(observation structs.Vector, k int) (class string) {
	knn := NewKnn(k, d.trainDataset)
	class = knn.PerformPrediction(&observation)
	return
}

func (d *Dataset) TestAlgorithm(k int) (result string) {
	expectedClasses := make([]string, len(d.trainDataset))
	actualClasses := make([]string, len(d.trainDataset))

	for i, vec := range d.trainDataset {
		expectedClasses[i] = vec.Class()
		actualClasses[i] = d.PredictWithKnn(vec, k)
	}

	accuracy := commons.MeasureAccuracy(expectedClasses, actualClasses)
	result = fmt.Sprintf("%d %", accuracy)
	return
}
