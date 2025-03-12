package structs

import (
	"fmt"
	"math"
)

type Vector struct {
	rows int32
	cols int32
	data [][]int32
	class string
}

func (v *Vector) Class() string {
    return v.class
}

func NewVector(data [][]int32, vecClass string) (*Vector, error) {
	rows := int32(len(data))
	if rows == 0 {
		return nil, fmt.Errorf("empty vector")
	}
	cols := int32(len(data[0]))

	return &Vector{
		rows: rows,
		cols: cols,
		data: data,
		class: vecClass,
	}, nil
}

func (v1 *Vector) AddVector(v2 *Vector) (*Vector, error) {
	if v1.rows != v2.rows || v1.cols != v2.cols {
		return nil, fmt.Errorf("cannot add vector with this dimensions: v1[%d, %d], v2[%d, %d]", v1.rows, v1.cols, v2.rows, v2.cols)
	}

	resultData := make([][]int32, v1.rows)

	for i := range v1.data {
		resultData[i] = make([]int32, v1.cols)
		for j := range v1.data[i] {
			resultData[i][j] = v1.data[i][j] + v2.data[i][j]
		}
	}

	return &Vector{
		rows: v1.rows,
		cols: v1.cols,
		data: resultData,
	}, nil
}

func (v1 *Vector) ScalarMultiply(scalar int32) *Vector {
	resultData := make([][]int32, v1.rows)

	for i := range v1.data {
		resultData[i] = make([]int32, v1.cols)
		for j := range v1.data[i] {
			resultData[i][j] = v1.data[i][j] * scalar
		}
	}

	return &Vector{
		rows: v1.rows,
		cols: v1.cols,
		data: resultData,
	}
}

func (v1 *Vector) EuclideanNorm() float64 {
	var sumSquares int32
	for i := range v1.data {
		for j := range v1.data[i] {
			sumSquares += v1.data[i][j] * v1.data[i][j]
		}
	}

	return math.Sqrt(float64(sumSquares))
}

func (v1 *Vector) EuclideanDistance(v2 *Vector) (float64, error) {
	if v1.rows != v2.rows || v1.cols != v2.cols {
		return 0, fmt.Errorf("cannot calculate distance for different dimensions: v1[%d, %d], v2[%d, %d]",
			v1.rows, v1.cols, v2.rows, v2.cols)
	}

	var sumSquares int32
	for i := range v1.data {
		for j := range v1.data[i] {
			diff := v1.data[i][j] - v2.data[i][j]
			sumSquares += diff * diff
		}
	}

	return math.Sqrt(float64(sumSquares)), nil
}

func (v1 *Vector) DotProduct(v2 *Vector) (int32, error) {
	if v1.rows != v2.rows || v1.cols != v2.cols {
		return 0, fmt.Errorf("cannot calculate dot product for different dimensions: v1[%d, %d], v2[%d, %d]",
			v1.rows, v1.cols, v2.rows, v2.cols)
	}

	var dot int32
	for i := range v1.data {
		for j := range v1.data[i] {
			dot += v1.data[i][j] * v2.data[i][j]
		}
	}

	return dot, nil
}

func (v1 *Vector) PrintVector() {
	for _, row := range v1.data {
		fmt.Println(row)
	}
}
