package structs

import (
	"math"
	"testing"
)

func TestNewVector(t *testing.T) {
	data := [][]int32{{1, 2}, {3, 4}}
	vec, err := NewVector(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if vec.rows != 2 || vec.cols != 2 {
		t.Fatalf("unexpected vector dimensions: got [%d,%d], expected [2,2]", vec.rows, vec.cols)
	}

	_, err = NewVector([][]int32{})
	if err == nil {
		t.Fatal("expected error for empty vector, got nil")
	}
}

func TestAddVector(t *testing.T) {
	v1, _ := NewVector([][]int32{{1, 2}, {3, 4}})
	v2, _ := NewVector([][]int32{{5, 6}, {7, 8}})

	result, err := v1.AddVector(v2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := [][]int32{{6, 8}, {10, 12}}

	for i := range expected {
		for j := range expected[i] {
			if result.data[i][j] != expected[i][j] {
				t.Errorf("wrong sum at [%d,%d]: got %d, expected %d", i, j, result.data[i][j], expected[i][j])
			}
		}
	}

	v3, _ := NewVector([][]int32{{1, 2, 3}})
	_, err = v1.AddVector(v3)
	if err == nil {
		t.Fatal("expected dimension mismatch error, got nil")
	}
}

func TestScalarMultiply(t *testing.T) {
	v1, _ := NewVector([][]int32{{1, 2}, {3, 4}})
	result := v1.ScalarMultiply(2)

	expected := [][]int32{{2, 4}, {6, 8}}

	for i := range expected {
		for j := range expected[i] {
			if result.data[i][j] != expected[i][j] {
				t.Errorf("wrong scalar multiply at [%d,%d]: got %d, expected %d", i, j, result.data[i][j], expected[i][j])
			}
		}
	}
}

func TestEuclideanNorm(t *testing.T) {
	v1, _ := NewVector([][]int32{{3, 4}})
	norm := v1.EuclideanNorm()

	expected := 5.0 // sqrt(3^2 + 4^2)
	if math.Abs(norm-expected) > 1e-6 {
		t.Errorf("unexpected norm: got %f, expected %f", norm, expected)
	}
}

func TestEuclideanDistance(t *testing.T) {
	v1, _ := NewVector([][]int32{{1, 2}})
	v2, _ := NewVector([][]int32{{4, 6}})

	dist, err := v1.EuclideanDistance(v2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := 5.0
	if math.Abs(dist-expected) > 1e-6 {
		t.Errorf("unexpected distance: got %f, expected %f", dist, expected)
	}

	v3, _ := NewVector([][]int32{{1}, {2}})
	_, err = v1.EuclideanDistance(v3)
	if err == nil {
		t.Fatal("expected dimension mismatch error, got nil")
	}
}

func TestDotProduct(t *testing.T) {
	v1, _ := NewVector([][]int32{{1, 2}})
	v2, _ := NewVector([][]int32{{3, 4}})

	dot, err := v1.DotProduct(v2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := int32(11)
	if dot != expected {
		t.Errorf("unexpected dot product: got %d, expected %d", dot, expected)
	}

	v3, _ := NewVector([][]int32{{1}, {2}})
	_, err = v1.DotProduct(v3)
	if err == nil {
		t.Fatal("expected dimension mismatch error, got nil")
	}
}
