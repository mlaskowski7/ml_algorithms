package main

import (
	"fmt"
	"log"
	"naiTasks/bayes"
	"naiTasks/commons"
)

func main() {
	dataset, err := commons.LoadDatasetFromCsv("data/bayesDataset.csv")
	if err != nil {
		log.Fatal(err)
		return
	}

	trainSet, testSet := commons.ClassificationTrainTestSplit(dataset, 2)
	fmt.Printf("trainSet: %v\n", trainSet)
	fmt.Printf("testSet: %v\n", testSet)

	nbc := bayes.NewNaiveBayesClassifier(true, trainSet)

	var expectedClasses []string
	var actualClasses []string

	for _, val := range testSet {
		expected := val["label"]
		predicted := nbc.Predict(val)

		expectedClasses = append(expectedClasses, expected)
		actualClasses = append(actualClasses, predicted)
	}

	accuracy := commons.MeasureAccuracy(expectedClasses, actualClasses)
	fmt.Printf("Model accuracy: %d%%\n", accuracy)

	classNames := findUniqueClasses(expectedClasses, actualClasses)

	for _, className := range classNames {
		tp := 0
		fp := 0
		fn := 0

		for i := 0; i < len(expectedClasses); i++ {
			trueLabel := expectedClasses[i]
			predLabel := actualClasses[i]

			if predLabel == className && trueLabel == className {
				tp++
			} else if predLabel == className && trueLabel != className {
				fp++
			} else if predLabel != className && trueLabel == className {
				fn++
			}
		}

		precision := commons.Precision(tp, fp)
		recall := commons.Recall(tp, fn, fp)
		f1 := commons.Fmeasure(int(precision*100), int(recall*100))

		fmt.Printf("Class '%s' - Precision: %.2f, Recall: %.2f, F1-score: %.2f\n", className, precision, recall, f1)
	}
}

func findUniqueClasses(expected, predicted []string) []string {
	classSet := make(map[string]bool)
	for _, label := range expected {
		classSet[label] = true
	}
	for _, label := range predicted {
		classSet[label] = true
	}

	var classes []string
	for label := range classSet {
		classes = append(classes, label)
	}
	return classes
}
