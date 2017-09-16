package knn

import (
	"container/heap"
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

// Neighbor provides storage for the parts we care about for knn.
type Neighbor struct {
	distance float64
	class    int
	index    int
}

// Neighbors implements a heap.Interface to store/sort the nearest neighbors,
// allowing for simpler/faster classification
type Neighbors []*Neighbor

// Len is required for sort.Interface.
func (n Neighbors) Len() int {
	return len(n)
}

// Less is required for sort.Interface.
func (n Neighbors) Less(i, j int) bool {
	return n[i].distance > n[j].distance
}

// Swap is required for sort.Interface.
func (n Neighbors) Swap(i, j int) {
	n[i], n[j] = n[j], n[i]
	n[i].index = i
	n[j].index = j
}

// Push implements heap.Interface.  Adds the neighbor to the correct place
// in the heap, assuring that the longest distance is ready to be popped.
func (n *Neighbors) Push(x interface{}) {
	neighbor := x.(*Neighbor)
	neighbor.index = len(*n)
	*n = append(*n, neighbor)
}

// Pop implementing heap.Interface. Removes and returns the Neighbor in the
// nearest neighbors set with the longest distance, effectively the farthest
// nearest neighbor.
func (n *Neighbors) Pop() interface{} {
	old := *n
	neighbor := old[len(old)-1]
	neighbor.index = -1 // for safety
	*n = old[:len(old)-1]
	return neighbor
}

func (n Neighbors) String() string {
	var s string
	for i, v := range n {
		s += fmt.Sprintf("i: %d\tNeighbor: %v\n", i, v)
	}
	return s
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

	//	distances := calculateDistances(d)

	for i := range d.Rows {
		neighbors := findNeighbors(i, k, d)
		predictions[i] = predictClass(neighbors, d)
	}

	return predictions
}

//func findNeighbors(i, k int, dist sparseDistance, d Dataset) map[int]bool {
func findNeighbors(i, k int, d Dataset) Neighbors {
	neighbors := make(Neighbors, 0)
	heap.Init(&neighbors)
	for j := range d.Rows {
		if j != i {
			dist := euclideanDistance(d.Rows[i], d.Rows[j])
			if neighbors.Len() < k {
				heap.Push(&neighbors, &Neighbor{distance: dist, class: d.GetClass(j)})
			} else if neighbors[0].distance >= dist {
				_ = heap.Pop(&neighbors)
				heap.Push(&neighbors, &Neighbor{distance: dist, class: d.GetClass(j)})
			}
		}
	}
	return neighbors
}

func predictClass(neighbors Neighbors, d Dataset) int {
	classes := make(map[int]int)
	for _, n := range neighbors {
		classes[n.class]++
	}
	pred := 0
	for class, count := range classes {
		if count > pred {
			pred = class
		}
	}
	return pred
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
