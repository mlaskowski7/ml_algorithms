package commons

import (
	"math/rand"
	"naiTasks/structs"
)

const trainSetPercent = 0.66

func TrainTestSplit(dataset *[]structs.Vector) ([]structs.Vector, []structs.Vector) {
	dereferencedDataset := *dataset
	rand.Shuffle(len(dereferencedDataset), func(i, j int) {
		dereferencedDataset[i], dereferencedDataset[j] = dereferencedDataset[j], dereferencedDataset[i]
	})

	trainSetLen := int(trainSetPercent * float64(len(dereferencedDataset)+1))

	return dereferencedDataset[:trainSetLen], dereferencedDataset[trainSetLen+1:]
}
