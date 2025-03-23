package main

import (
	"fmt"
	"naiTasks/perceptron"
)

func main() {
	err := perceptron.TrainPerceptronOnIrisCsv("data/iris.csv")
	if err != nil {
		fmt.Println(err)
	}
}
