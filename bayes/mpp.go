package bayes

type NaiveBayesClassifier struct {
	applySmoothingAll bool
	trainDataset      [][]float64
}
