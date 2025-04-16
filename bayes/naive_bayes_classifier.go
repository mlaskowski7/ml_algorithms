package bayes

func SimpleSmoothing(numerator int, denominator int, numberOfClasses int) float64 {
	result := float64(numerator+1) / float64(denominator+numberOfClasses)
	return result
}
