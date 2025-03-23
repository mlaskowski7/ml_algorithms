package structs

import (
	"fmt"
	"math"
)

type Vector struct {
	rows  int32
	cols  int32
	data  [][]float64
	class string
}

func (v *Vector) Class() string {
	return v.class
}

func (v *Vector) SetClass(class string) {
	v.class = class
}

func NewVector(data [][]float64, vecClass string) (*Vector, error) {
	rows := int32(len(data))
	if rows == 0 {
		return nil, fmt.Errorf("empty vector")
	}
	cols := int32(len(data[0]))

	return &Vector{
		rows:  rows,
		cols:  cols,
		data:  data,
		class: vecClass,
	}, nil
}

func (v *Vector) AddVector(v2 *Vector) (*Vector, error) {
	if v.rows != v2.rows || v.cols != v2.cols {
		return nil, fmt.Errorf("cannot add vector with this dimensions: v1[%d, %d], v2[%d, %d]", v.rows, v.cols, v2.rows, v2.cols)
	}

	resultData := make([][]float64, v.rows)

	for i := range v.data {
		resultData[i] = make([]float64, v.cols)
		for j := range v.data[i] {
			resultData[i][j] = v.data[i][j] + v2.data[i][j]
		}
	}

	return &Vector{
		rows: v.rows,
		cols: v.cols,
		data: resultData,
	}, nil
}

func (v *Vector) ScalarMultiply(scalar float64) *Vector {
	resultData := make([][]float64, v.rows)

	for i := range v.data {
		resultData[i] = make([]float64, v.cols)
		for j := range v.data[i] {
			resultData[i][j] = v.data[i][j] * scalar
		}
	}

	return &Vector{
		rows: v.rows,
		cols: v.cols,
		data: resultData,
	}
}

func (v *Vector) EuclideanNorm() float64 {
	var sumSquares float64
	for i := range v.data {
		for j := range v.data[i] {
			sumSquares += v.data[i][j] * v.data[i][j]
		}
	}

	return math.Sqrt(sumSquares)
}

func (v *Vector) EuclideanDistance(v2 *Vector) (float64, error) {
	if v.rows != v2.rows || v.cols != v2.cols {
		return 0, fmt.Errorf("cannot calculate distance for different dimensions: v1[%d, %d], v2[%d, %d]",
			v.rows, v.cols, v2.rows, v2.cols)
	}

	var sumSquares float64
	for i := range v.data {
		for j := range v.data[i] {
			diff := v.data[i][j] - v2.data[i][j]
			sumSquares += diff * diff
		}
	}

	return math.Sqrt(sumSquares), nil
}

func (v *Vector) DotProduct(v2 *Vector) (float64, error) {
	if v.rows != v2.rows || v.cols != v2.cols {
		return 0, fmt.Errorf("cannot calculate dot product for different dimensions: v1[%d, %d], v2[%d, %d]",
			v.rows, v.cols, v2.rows, v2.cols)
	}

	var dot float64
	for i := range v.data {
		for j := range v.data[i] {
			dot += v.data[i][j] * v2.data[i][j]
		}
	}

	return dot, nil
}

func (v *Vector) PrintVector() {
	fmt.Printf("class: %s", v.class)
	for _, row := range v.data {
		fmt.Println(row)
	}
}
