package knn

import (
	"fmt"
	"math"

	"github.com/bsm/arff"
)

// Dataset wraps arff.DataRow to provide a little more information
type Dataset struct {
	Rows []arff.DataRow

	numClasses int
}

// NewDataset constructs a Dataset from the data contained in the provided
// file name.  Currently panics if anything untoward happens.
func NewDataset(fname string) Dataset {
	f, err := arff.Open(fname)
	if err != nil {
		panic("failed to open file: " + err.Error())
	}
	defer f.Close()

	rows, err := f.ReadAll()
	if err != nil {
		panic("failed to read file: " + err.Error())
	}
	return Dataset{Rows: rows}
}

// NumClasses returns (and possibly calculates) the number of classes present
// in the Dataset
func (d *Dataset) NumClasses() int {
	if d.numClasses != 0 {
		return d.numClasses
	}

	for _, row := range d.Rows {
		class := int(row.Values[len(row.Values)-1].(float64))
		if class > d.numClasses {
			d.numClasses = class
		}
	}

	d.numClasses++ // 0 is a possible class
	return d.numClasses
}

// GetClass returns the class of the instance at index i in the dataset
func (d Dataset) GetClass(i int) int {
	r := d.Rows[i]
	return int(r.Values[len(r.Values)-1].(float64))
}

type sparseDistance [][]float64

func (s sparseDistance) fetchDistance(i, j int) float64 {
	if j < i {
		i, j = j, i
	}
	return s[i][j]
}

// ReadARFF is just me playing around
func ReadARFF() {
	data, err := arff.Open("./samples/small.arff")
	if err != nil {
		panic("failed to open file: " + err.Error())
	}
	defer data.Close()

	for data.Next() {
		fmt.Println(data.Row().Values...)
	}
	if err := data.Err(); err != nil {
		panic("failed to read file: " + err.Error())
	}
}

// KNN implements the K Nearest Neighbors algorithm to predict
// the class of the particular DataRow based on the class of the
// k nearest neighbors.
func KNN(k int, d Dataset) []int {
	predictions := make([]int, len(d.Rows))

	distances := calculateDistances(d)

	for i := range d.Rows {
		neighbors := findNeighbors(i, k, distances, d)
		predictions[i] = predictClass(neighbors, d)
	}

	return predictions
}

func findNeighbors(i, k int, dist sparseDistance, d Dataset) map[int]bool {
	neighbors := make(map[int]bool)
	for len(neighbors) < k {
		min := -1
		for j := 0; j < len(d.Rows); j++ { // find the minimum distance
			if j != i && !neighbors[j] {
				if min == -1 {
					min = j
				}
				if dist.fetchDistance(i, j) < dist.fetchDistance(i, min) {
					min = j
				}
			}
		}
		neighbors[min] = true
	}
	return neighbors
}

func predictClass(neighbors map[int]bool, d Dataset) int {
	classes := make(map[int]int)
	for n := range neighbors {
		classes[d.GetClass(n)]++
	}
	pred := 0
	for class, count := range classes {
		if count > pred {
			pred = class
		}
	}
	return pred
}

func calculateDistances(d Dataset) sparseDistance {
	distances := make(sparseDistance, len(d.Rows))

	for i := range d.Rows {
		distances[i] = make([]float64, len(d.Rows))
		for j := len(d.Rows) - 1; j > i; j-- {
			distances[i][j] = euclideanDistance(d.Rows[i], d.Rows[j])
		}
	}

	return distances
}

func euclideanDistance(a, b arff.DataRow) float64 {
	var squares float64
	for i := range a.Values[:len(a.Values)-1] {
		diff := a.Values[i].(float64) - b.Values[i].(float64)
		squares += diff * diff
	}
	return math.Sqrt(squares)
}

// ComputeConfusionMatrix calculates the level of confusion experienced by
// our KNN algorithm
func ComputeConfusionMatrix(pred []int, d Dataset) [][]int {
	// allocate the matrix based on the number of classes
	confusionMatrix := make([][]int, d.NumClasses())
	for i := range confusionMatrix {
		confusionMatrix[i] = make([]int, d.NumClasses())
	}

	for i := range d.Rows {
		row := d.Rows[i]
		trueClass := int(row.Values[len(row.Values)-1].(float64))
		predictedClass := pred[i]

		// if the predictedClass and the trueClass match, then they
		// add up on the diagonal.
		confusionMatrix[trueClass][predictedClass]++
	}

	return confusionMatrix
}

// ComputeAccuracy determines just how accurate our algorithm was
func ComputeAccuracy(confusion [][]int, d Dataset) float64 {
	var correctPredictions int

	for i := range confusion {
		correctPredictions += confusion[i][i]
	}

	return float64(correctPredictions) / float64(len(d.Rows))
}
