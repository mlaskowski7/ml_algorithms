package main

import (
	"fmt"
	"naiTasks/bayes"
)

func main() {
	//_, _ = singlelayernn.TestLanguageNeuralNetwork()

	fmt.Println("after smoothing ", bayes.SimpleSmoothing(0, 10, 12))
}
