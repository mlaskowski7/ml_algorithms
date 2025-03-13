package knn

import (
	"encoding/csv"
	"fmt"
	"naiTasks/commons"
	"naiTasks/structs"
	"os"
	"strconv"
)

func performKnnFromCsv(filename string) (string, error) {
	file, err := os.Open("data/" + filename)
	if err != nil {
		return "", err
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
		return "", err
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

	fmt.Printf("Train Set Size: %d, Test Set Size: %d\n", len(trainDataSet), len(testDataSet))
	return "", nil

}
