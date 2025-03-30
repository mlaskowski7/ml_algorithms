package commons

import "fmt"

// MeasureAccuracy returns percent of correct predictions, example: returned 100 = 100% prediction rate
func MeasureAccuracy[T comparable](expectedClasses []T, actualClasses []T) int {
	if len(expectedClasses) != len(actualClasses) {
		panic(fmt.Sprintf("Length mismatch between expectedClasses(%d) and actualClasses(%d)", len(expectedClasses), len(actualClasses)))
	}

	var correctCounter int
	for i := 0; i < len(expectedClasses); i++ {
		if expectedClasses[i] == actualClasses[i] {
			correctCounter++
		}
	}

	return int(float64(correctCounter) / float64(len(actualClasses)) * 100)
}

func Fmeasure(p int, r int) float64 {
	return float64(2*p*r) / float64(p+r)
}

func Recall(tp int, fn int, fp int) float64 {
	return float64(tp) / float64(fn+fp)
}

func Precision(tp int, fp int) float64 {
	return float64(tp) / float64(fp+tp)
}
