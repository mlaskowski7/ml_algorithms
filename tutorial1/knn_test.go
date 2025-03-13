package knn

import (
	"naiTasks/structs"
	"testing"
)


func TestNewKnn(t *testing.T) {
	testTrainingSet := []structs.Vector {
		*newVectorHelper([][]int32{{1, 2}}, "A"),
		*newVectorHelper([][]int32{{3, 4}}, "B"),
	}

	knn := NewKnn(2, testTrainingSet)

	k := knn.k
	if k != 2 {
		t.Errorf("Expected k to be 2, but was %d", k)
	}

	trainDataSize := len(knn.trainDataset)
	if trainDataSize != 2 {
		t.Errorf("Expected train data size to be 2, but was %d", trainDataSize)
	}
}

func TestCalculateDistances(t *testing.T) {
	data := []structs.Vector{
		*newVectorHelper([][]int32{{1, 1}}, "A"),
		*newVectorHelper([][]int32{{4, 4}}, "B"),
	}

	knn := NewKnn(2, data)
	testVec := newVectorHelper([][]int32{{2, 2}}, "C")

	knn.calculateDistances(testVec)

	if len(knn.distances) != 2 {
		t.Errorf("Expected distances map size to be 2, got %d", len(knn.distances))
	}

	if int(knn.distances["A"][0]) != 1 && int(knn.distances["B"][0]) != 3 {
		t.Error("Expected calculated distances are uncorrect")
	}
}

func newVectorHelper(data [][]int32, class string) 	*structs.Vector {
	v, err := structs.NewVector(data, class)
	if err != nil {
		panic(err)
	}

	return v
}