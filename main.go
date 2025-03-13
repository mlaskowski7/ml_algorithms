package main

import "naiTasks/knn"

func main() {
	_, err := knn.NewDatasetFromCsv("IRIS.csv")
	if err != nil {
		return
	}
}
