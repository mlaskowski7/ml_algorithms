package bayes

import (
	"fmt"
	"math"
)

type NaiveBayesClassifier struct {
	applySmoothingAll            bool
	labelCountsMap               map[string]int
	trainDatasetSize             int
	headerValueForLabelCountsMap map[string]map[string]map[string]int
	headerValuesMap              map[string]map[string]bool
}

func NewNaiveBayesClassifier(applySmoothingAll bool, trainDataset []map[string]string) *NaiveBayesClassifier {
	nbc := &NaiveBayesClassifier{
		applySmoothingAll:            applySmoothingAll,
		labelCountsMap:               make(map[string]int),
		trainDatasetSize:             len(trainDataset),
		headerValueForLabelCountsMap: make(map[string]map[string]map[string]int),
		headerValuesMap:              make(map[string]map[string]bool),
	}

	nbc.train(trainDataset)
	return nbc
}

func (nbc *NaiveBayesClassifier) train(trainDataset []map[string]string) {
	for _, dataRow := range trainDataset {
		label := dataRow["label"]
		nbc.labelCountsMap[label]++

		for header, value := range dataRow {
			if header != "label" {
				if nbc.headerValueForLabelCountsMap[header] == nil {
					nbc.headerValueForLabelCountsMap[header] = make(map[string]map[string]int)
				}
				if nbc.headerValueForLabelCountsMap[header][value] == nil {
					nbc.headerValueForLabelCountsMap[header][value] = make(map[string]int)
				}
				nbc.headerValueForLabelCountsMap[header][value][label]++
				if nbc.headerValuesMap[header] == nil {
					nbc.headerValuesMap[header] = make(map[string]bool)
				}
				nbc.headerValuesMap[header][value] = true
			}
		}
	}
	nbc.printPrioriProbabilities()
	nbc.printPosterioriProbabilities()
}

func (nbc *NaiveBayesClassifier) Predict(input map[string]string) string {
	maxLabel := ""
	maxProb := math.Inf(-1)

	for label := range nbc.labelCountsMap {
		labelProb := math.Log(float64(nbc.labelCountsMap[label]) / float64(nbc.trainDatasetSize))

		for header, val := range input {
			valCount := 0
			totalCountForHeader := 0

			if nbc.headerValueForLabelCountsMap[header] != nil {
				if nbc.headerValueForLabelCountsMap[header][val] != nil {
					valCount = nbc.headerValueForLabelCountsMap[header][val][label]
				}
				for v := range nbc.headerValuesMap[header] {
					totalCountForHeader += nbc.headerValueForLabelCountsMap[header][v][label]
				}
			}

			// smoothing
			prob := 0.0
			numClasses := len(nbc.headerValuesMap[header])
			if nbc.applySmoothingAll {
				prob = SimpleSmoothing(valCount, totalCountForHeader, numClasses)
			}

			labelProb += math.Log(prob)
		}

		if labelProb > maxProb {
			maxProb = labelProb
			maxLabel = label
		}
	}

	return maxLabel
}

func (nbc *NaiveBayesClassifier) printPrioriProbabilities() {
	fmt.Println("Calculated A Priori probabilities:")
	for label, count := range nbc.labelCountsMap {
		aPrioriProb := float32(count) / float32(nbc.trainDatasetSize)
		fmt.Printf("Probability for %s = %.2f\n", label, aPrioriProb)
	}
}

func (nbc *NaiveBayesClassifier) printPosterioriProbabilities() {
	fmt.Println("Calculated A Posteriori probabilities:")
	for header, valueMap := range nbc.headerValueForLabelCountsMap {
		fmt.Printf("For header = %s:\n", header)
		for value, labelsMap := range valueMap {
			for label, count := range labelsMap {
				total := 0
				for val := range nbc.headerValueForLabelCountsMap[header] {
					total += nbc.headerValueForLabelCountsMap[header][val][label]
				}
				numClasses := len(nbc.headerValuesMap[header])

				var prob float64
				if nbc.applySmoothingAll {
					prob = SimpleSmoothing(count, total, numClasses)
				} else {
					if count == 0 {
						prob = SimpleSmoothing(count, total, numClasses)
					} else {
						prob = float64(count) / float64(total)
					}
				}

				fmt.Printf("  P(%s=%s | %s) = %.4f\n", header, value, label, prob)
			}
		}
	}
}
