package knn

import (
	"testing"
)

func TestNewDatasetFromCsv(t *testing.T) {
	dataset, err := NewDatasetFromCsv("../data/IRIS.csv")
	if err != nil {
		t.Fatalf("Failed to create new dataset from csv file -> %s", err.Error())
	}

	if len(dataset.TrainDataset) != 100 {
		t.Errorf("Expected the train dataset to be length of 100, but was %d", len(dataset.TrainDataset))
	}
}

func TestPredictWithKnn(t *testing.T) {
	dataset, _ := NewDatasetFromCsv("../data/IRIS.csv")
	if len(dataset.TrainDataset) != 100 {
		t.Fatalf("Training data set was expected to be 100 observations, but was %d", len(dataset.TrainDataset))
	}

	observation := dataset.TestDataset[0]
	predictedClass, err := dataset.PredictWithKnn(observation, 3)

	if err != nil {
		t.Fatalf("PredictWithKnn() failed: %s", err.Error())
	}

	if predictedClass == "" {
		t.Fatalf("Predicted class should not be empty")
	}
}
