package knn

import (
	"fmt"
	"testing"
)

//func TestReadARFF(t *testing.T) {
//	ReadARFF()
//}

func TestSmallData(t *testing.T) {
	d := NewDataset("./samples/small.arff")

	p := KNN(3, d)
	c := ComputeConfusionMatrix(p, d)
	a := ComputeAccuracy(c, d)

	fmt.Printf("The KNN classifier for %d instances required %d ms CPU time, accuracy was %.4f\n", len(d.Rows), 0, a)
}

func TestMediumData(t *testing.T) {
	d := NewDataset("./samples/medium.arff")

	p := KNN(3, d)
	c := ComputeConfusionMatrix(p, d)
	a := ComputeAccuracy(c, d)

	fmt.Printf("The KNN classifier for %d instances required %d ms CPU time, accuracy was %.4f\n", len(d.Rows), 0, a)
}

func BenchmarkSmall(b *testing.B) {
	d := NewDataset("./samples/small.arff")

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		_ = KNN(3, d)
	}
	//fmt.Println(dist)
}

func BenchmarkMedium(b *testing.B) {
	d := NewDataset("./samples/medium.arff")

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		_ = KNN(3, d)
	}
	//fmt.Println(dist)
}
