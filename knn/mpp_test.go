package knn

import (
	"testing"
)

func TestNewDatasetFromCsv(t *testing.T) {
	dataset, err := NewDatasetFromCsv("IRIS.csv")
	if err != nil {
		t.Fatalf("Failed to create new dataset from csv file -> %s", err.Error())
	}

	if len(dataset.trainDataset)+len(dataset.testDataset) != 150 {
		t.Errorf("Expected the divided daataset to be in sum 150, but was %d", len(dataset.trainDataset)+len(dataset.testDataset))
	}

	if len(dataset.trainDataset) != 100 {
		t.Errorf("Expected the train dataset to be length of 100, but was %d", len(dataset.trainDataset))
	}
}

func TestPredictWithKnn(t *testing.T) {
	dataset, _ := NewDatasetFromCsv("../data/IRIS.csv")
	if len(dataset.trainDataset) != 100 {
		t.Fatalf("Training data set was expected to be 100 observations, but was %d", len(dataset.trainDataset))
	}

	observation := dataset.testDataset[0]
	predictedClass := dataset.PredictWithKnn(observation, 3)

	if predictedClass == "" {
		t.Fatalf("Predicted class should not be empty")
	}
}
