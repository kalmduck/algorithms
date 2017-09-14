package articulation

import (
	"errors"
	"log"

	"github.com/kalmduck/container/graph"
)

// This package implements Tarjan's Algorithm for determining the articulation
// points (if any) in an undirected graph.

type tracker struct {
	visited []bool
	depth   []int
	low     []int
	parent  []int
	ap      []int
}

func newTracker(n int) *tracker {
	t := &tracker{
		make([]bool, n),
		make([]int, n),
		make([]int, n),
		make([]int, n),
		nil,
	}
	for i := range t.parent {
		t.parent[i] = -1
	}
	return t
}

// GetArticulationPoints returns a slice containing the nodes
// determined to be articulation points.
func GetArticulationPoints(g *graph.Graph) []int {
	t := newTracker(g.Size())
	for !t.allVisited() {
		n, err := t.nextUnvisited()
		if err != nil {
			log.Fatal("Still looping when all have been visited")
		}
		findAP(g, t, n, 0)
	}
	return t.ap
}

// findAP performs the actual work of finding the articulation points in a
// connected graph.
func findAP(g *graph.Graph, t *tracker, i, d int) {
	t.visited[i] = true
	t.depth[i] = d
	t.low[i] = d
	childCount := 0
	isAP := false

	// simple min function for ints.
	min := func(l, ladj int) int {
		if l < ladj {
			return l
		}
		return ladj
	}
	n := g.GetNode(i)
	for adj := range n.Edges {
		if !t.visited[adj] { // we haven't been hear yet
			t.parent[adj] = i // mark the parent so we don't go in reverse
			findAP(g, t, adj, d+1)
			childCount = childCount + 1

			// adj doesn't have any loops to an ancestor
			if t.low[adj] >= t.depth[i] {
				isAP = true
			}

			// update the lowpoint based on this child
			t.low[i] = min(t.low[i], t.low[adj])
		} else if adj != t.parent[i] { // already visited, but not the direct parent
			t.low[i] = min(t.low[i], t.depth[adj])
		}
	}
	// non-root and isAP or root and at least 2 children
	if (t.parent[i] >= 0 && isAP) || (t.parent[i] == -1 && childCount > 1) {
		t.ap = append(t.ap, i)
	}
}

func (t tracker) nextUnvisited() (int, error) {
	for i, v := range t.visited {
		if !v {
			return i, nil
		}
	}
	return -1, errors.New("nextUnvisited() called on tracker with none left to visit")
}

func (t tracker) allVisited() bool {
	done := true
	for _, v := range t.visited {
		done = done && v
	}
	return done
}
