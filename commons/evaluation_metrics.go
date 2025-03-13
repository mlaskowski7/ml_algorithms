package commons

// MeasureAccuracy returns percent of correct predictions, example: returned 100 = 100% prediction rate
func MeasureAccuracy(expectedClasses []string, actualClasses []string) int {
	var correctCounter int
	for i := 0; i < len(actualClasses); i++ {
		if actualClasses[i] == expectedClasses[i] {
			correctCounter++
		}
	}

	return int(correctCounter/len(actualClasses)) * 100
}
