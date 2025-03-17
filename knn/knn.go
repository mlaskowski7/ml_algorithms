package knn

import (
	"fmt"
	"naiTasks/structs"
)

type KNearestNeighbours struct {
	k            int
	trainDataset []structs.Vector
}

type Neighbour struct {
	Class    string
	Distance float64
}

func NewKnn(k int, trainDataset []structs.Vector) *KNearestNeighbours {
	return &KNearestNeighbours{
		k:            k,
		trainDataset: trainDataset,
	}
}

func (knn *KNearestNeighbours) PerformPrediction(vec *structs.Vector) (string, error) {
	if len(knn.trainDataset) < knn.k {
		return "", fmt.Errorf(" The training dataset is too short, the length is %d and k = %d", len(knn.trainDataset), knn.k)
	}

	neighbours := knn.calculateDistances(vec)
	votes := knn.calculateVotes(neighbours)

	return knn.finalPrediction(votes), nil
}

func (knn *KNearestNeighbours) calculateVotes(neighbours []Neighbour) map[string]int {
	votes := make(map[string]int)
	for i := 0; i < knn.k; i++ {
		votes[neighbours[i].Class]++
	}

	return votes
}

func (knn *KNearestNeighbours) finalPrediction(votes map[string]int) string {
	var predictedClass string
	maxVotes := -1
	for class, count := range votes {
		if count > maxVotes {
			maxVotes = count
			predictedClass = class
		}
	}

	return predictedClass
}

func (knn *KNearestNeighbours) calculateDistances(vec *structs.Vector) []Neighbour {
	var neighbours []Neighbour

	for _, observation := range knn.trainDataset {
		dist, err := observation.EuclideanDistance(vec)
		if err != nil {
			continue
		}

		neighbours = append(neighbours, Neighbour{
			Class:    observation.Class(),
			Distance: dist,
		})
	}

	return knn.insertionSort(neighbours)
}

func (knn *KNearestNeighbours) insertionSort(arr []Neighbour) []Neighbour {
	for i := 1; i < len(arr); i++ {
		key := arr[i]
		j := i - 1

		for j >= 0 && arr[j].Distance > key.Distance {
			arr[j+1] = arr[j]
			j--
		}
		arr[j+1] = key
	}
	return arr
}
