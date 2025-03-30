package singlelayernn

import "strings"

func TransformStringToSliceWithLettersCount(text string) []float64 {
	result := make([]float64, 26)
	text = strings.ToLower(text)
	letters := 0

	for _, c := range text {
		if c >= 'a' && c <= 'z' {
			index := c - 'a'
			result[index]++
			letters++
		}
	}

	return result
}
