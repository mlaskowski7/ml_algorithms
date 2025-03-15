package knn

import (
	"naiTasks/structs"
)

type KNearestNeighbours struct {
	k            int
	trainDataset []structs.Vector
	distances    map[string][]float64
	modes        map[string]float64
}

func NewKnn(k int, trainDataset []structs.Vector) *KNearestNeighbours {
	return &KNearestNeighbours{
		k:            k,
		trainDataset: trainDataset,
		distances:    make(map[string][]float64),
		modes:        make(map[string]float64),
	}
}

func (knn *KNearestNeighbours) PerformPrediction(vec *structs.Vector) string {
	knn.calculateDistances(vec)
	knn.sortDistances()
	knn.calculateModes()
	return knn.getMinMode()
}

func (knn *KNearestNeighbours) calculateDistances(vec *structs.Vector) {
	for i := 0; i < knn.k; i++ {
		observation := knn.trainDataset[len(knn.trainDataset)-1-i]
		class := observation.Class()

		if _, ok := knn.distances[class]; !ok {
			knn.distances[class] = make([]float64, 0)
		}

		euclideanDist, _ := observation.EuclideanDistance(vec)
		knn.distances[class] = append(knn.distances[class], euclideanDist)
	}
}

func (knn *KNearestNeighbours) sortDistances() {
	for class, distances := range knn.distances {
		knn.distances[class] = knn.insertionSort(distances)
	}
}

func (knn *KNearestNeighbours) calculateModes() {
	for class, distances := range knn.distances {
		frequencies := make(map[float64]int)

		for _, dist := range distances {
			frequencies[dist]++
		}

		var mostOccurring float64
		var maxCount int
		for dist, count := range frequencies {
			if count > maxCount {
				mostOccurring = dist
				maxCount = count
			}
		}

		knn.modes[class] = mostOccurring
	}
}

func (knn *KNearestNeighbours) getMinMode() string {
	var minKey string
	var minValue float64 = 1e9

	for key, value := range knn.modes {
		if value < minValue {
			minValue = value
			minKey = key
		}
	}

	return minKey
}

func (knn *KNearestNeighbours) insertionSort(arr []float64) []float64 {
	for i := 1; i < len(arr); i++ {
		key := arr[i]
		j := i - 1

		for j >= 0 && arr[j] > key {
			arr[j+1] = arr[j]
			j--
		}
		arr[j+1] = key
	}
	return arr
}
