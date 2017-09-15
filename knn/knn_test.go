package knn

import (
	"fmt"
	"math"
	"testing"
)

//func TestReadARFF(t *testing.T) {
//	ReadARFF()
//}

func TestSmallData(t *testing.T) {
	d := NewDataset("./samples/small.arff")

	p := KNN(4, d)
	c := ComputeConfusionMatrix(p, d)
	a := ComputeAccuracy(c, d)

	fmt.Printf("The KNN classifier for %d instances required %d ms CPU time, accuracy was %.4f\n", len(d.Rows), 0, a)
}

func TestMediumData(t *testing.T) {
	d := NewDataset("./samples/medium.arff")

	p := KNN(4, d)
	c := ComputeConfusionMatrix(p, d)
	a := ComputeAccuracy(c, d)

	fmt.Printf("The KNN classifier for %d instances required %d ms CPU time, accuracy was %.4f\n", len(d.Rows), 0, a)
}

func BenchmarkCalculateDistances(b *testing.B) {
	d := NewDataset("./samples/medium.arff")

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		_ = calculateDistances(d)
	}
	//fmt.Println(dist)
}

func BenchmarkMult(b *testing.B) {
	for n := 0; n < b.N; n++ {
		_ = 156.04 * 156.04
	}
}

func BenchmarkPow(b *testing.B) {
	for n := 0; n < b.N; n++ {
		_ = math.Pow(156.04, 2)
	}
}
