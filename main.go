package main

import (
	"fmt"
	"naiTasks/knn"
)

func main() {
	dataset, err := knn.NewDatasetFromCsv("data/IRIS.csv")
	if err != nil {
		fmt.Println(err)
		return
	}

	for i := 1; i < 10; i++ {
		accuracy, err := dataset.TestAlgorithm(i)
		if err == nil {
			fmt.Printf("Accuracy for k = %d is equal to %s\n", i, accuracy)
		}
	}
}
