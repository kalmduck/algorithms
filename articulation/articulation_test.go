package articulation

import (
	"fmt"
	"testing"

	"github.com/kalmduck/container/graph"
)

func TestAllVisited(t *testing.T) {
	track := newTracker(5)
	if track.allVisited() {
		t.Error("allVisited() should have returned false.")
	}
	for i := range track.visited {
		track.visited[i] = true
	}
	if !track.allVisited() {
		t.Error("allVisited() should have returned true.")
	}
}

func TestNextUnvisited(t *testing.T) {
	track := newTracker(5)
	if n, _ := track.nextUnvisited(); n != 0 {
		t.Error("Next unvisited wasn't 0")
	}
	track.visited[0] = true
	if n, _ := track.nextUnvisited(); n != 1 {
		t.Error("Next unvisited wasn't 1")
	}
	for i := range track.visited {
		track.visited[i] = true
	}
	if _, err := track.nextUnvisited(); err == nil {
		t.Error("Should have gotten an error for a fully visited graph")
	}
}

func TestVerySimpleGraph(t *testing.T) {
	g := graph.New(3)
	g.AddEdge(0, 1)
	g.AddEdge(1, 2)
	ap := GetArticulationPoints(g)
	if len(ap) != 1 {
		t.Errorf("len(ap): expected: 1, actual: %d\n", len(ap))
	}
	if ap[0] != 1 {
		t.Errorf("expected ap: 1, actual: %d\n", ap[0])
	}
}

func TestUnconnectedGraph(t *testing.T) {
	g := graph.New(6)
	g.AddEdge(0, 1)
	g.AddEdge(1, 2)
	g.AddEdge(3, 4)
	g.AddEdge(4, 5)
	ap := GetArticulationPoints(g)
	if len(ap) != 2 {
		t.Errorf("expected 2 ap nodes, got: %d", len(ap))
	}
	if !contains(ap, 1) {
		t.Error("ap doesn't have 1")
	}
	if !contains(ap, 4) {
		t.Error("ap doesn't have 4")
	}
}

func contains(vals []int, n int) bool {
	for _, v := range vals {
		if v == n {
			return true
		}
	}
	return false
}

func TestRandomGraph(t *testing.T) {
	g := graph.NewRandomDensityGraph(100, 50)
	fmt.Println(g)
	ap := GetArticulationPoints(g)
	fmt.Println(ap)
}
