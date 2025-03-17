package commons

import (
	"math"
	"math/rand"
	"naiTasks/structs"
)

const trainSetPercent = 0.66

func TrainTestSplit(dataset *[]structs.Vector) ([]structs.Vector, []structs.Vector) {
	dereferencedDataset := *dataset

	classGroups := make(map[string][]structs.Vector)
	for _, sample := range dereferencedDataset {
		classGroups[sample.Class()] = append(classGroups[sample.Class()], sample)
	}

	var trainSet []structs.Vector
	var testSet []structs.Vector

	for _, group := range classGroups {
		rand.Shuffle(len(group), func(i, j int) {
			group[i], group[j] = group[j], group[i]
		})

		trainLen := int(math.Round(trainSetPercent * float64(len(group))))

		trainSet = append(trainSet, group[:trainLen]...)
		testSet = append(testSet, group[trainLen:]...)
	}

	rand.Shuffle(len(trainSet), func(i, j int) {
		trainSet[i], trainSet[j] = trainSet[j], trainSet[i]
	})

	rand.Shuffle(len(testSet), func(i, j int) {
		testSet[i], testSet[j] = testSet[j], testSet[i]
	})

	return trainSet, testSet
}
