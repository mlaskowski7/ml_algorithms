package commons

// MeasureAccuracy returns percent of correct predictions, example: returned 100 = 100% prediction rate
func MeasureAccuracy[T comparable](expectedClasses []T, actualClasses []T) int {
	if len(expectedClasses) != len(actualClasses) {
		panic("Length mismatch between expectedClasses and actualClasses")
	}

	var correctCounter int
	for i := 0; i < len(expectedClasses); i++ {
		if expectedClasses[i] == actualClasses[i] {
			correctCounter++
		}
	}

	return int(float64(correctCounter) / float64(len(actualClasses)) * 100)
}
