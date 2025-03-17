package knn

import (
	"naiTasks/structs"
	"testing"
)

func TestNewKnn(t *testing.T) {
	testTrainingSet := []structs.Vector{
		*newVectorHelper([][]float64{{1, 2}}, "A"),
		*newVectorHelper([][]float64{{3, 4}}, "B"),
	}

	knn := NewKnn(2, testTrainingSet)

	if knn.k != 2 {
		t.Errorf("Expected k to be 2, but was %d", knn.k)
	}

	if len(knn.trainDataset) != 2 {
		t.Errorf("Expected train dataset length to be 2, but was %d", len(knn.trainDataset))
	}
}

func TestCalculateDistances(t *testing.T) {
	data := []structs.Vector{
		*newVectorHelper([][]float64{{1, 1}}, "A"),
		*newVectorHelper([][]float64{{4, 4}}, "B"),
	}

	knn := NewKnn(2, data)

	testVec := newVectorHelper([][]float64{{2, 2}}, "C")

	neighbours := knn.calculateDistances(testVec)

	if len(neighbours) != 2 {
		t.Errorf("Expected 2 neighbours, but got %d", len(neighbours))
	}

	expectedOrder := []string{"A", "B"}
	for i, neighbor := range neighbours {
		if neighbor.Class != expectedOrder[i] {
			t.Errorf("Expected class %s at index %d, but got %s", expectedOrder[i], i, neighbor.Class)
		}
	}

	expectedDistances := []float64{
		1.4142135623730951,
		2.8284271247461903,
	}

	for i, neighbor := range neighbours {
		if neighbor.Distance != expectedDistances[i] {
			t.Errorf("Expected distance %.4f, but got %.4f", expectedDistances[i], neighbor.Distance)
		}
	}
}

func TestPerformPrediction(t *testing.T) {
	data := []structs.Vector{
		*newVectorHelper([][]float64{{1, 1}}, "A"),
		*newVectorHelper([][]float64{{2, 2}}, "A"),
		*newVectorHelper([][]float64{{3, 3}}, "B"),
	}

	knn := NewKnn(2, data)

	testVec := newVectorHelper([][]float64{{2, 2}}, "C")

	predicted, err := knn.PerformPrediction(testVec)
	if err != nil {
		t.Fatalf("PerformPrediction returned an error: %v", err)
	}

	if predicted != "A" {
		t.Errorf("Expected predicted class to be 'A', but got '%s'", predicted)
	}
}

func newVectorHelper(data [][]float64, class string) *structs.Vector {
	v, err := structs.NewVector(data, class)
	if err != nil {
		panic(err)
	}
	return v
}
