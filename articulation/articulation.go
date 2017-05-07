package articulation

import "github.com/kalmduck/container/graph"

// This package implements Tarjan's Algorithm for determining the articulation
// points (if any) in an undirected graph.

var visited []bool
var depth []int
var low []int
var parent []int
var ap []int

// GetArticulationPoints returns a slice containing the nodes
// determined to be articulation points.
func GetArticulationPoints(g *graph.Graph) []int {
	visited = make([]bool, len(g.Nodes))
	depth = make([]int, len(g.Nodes))
	low = make([]int, len(g.Nodes))
	parent = make([]int, len(g.Nodes))
	for i := range parent {
		parent[i] = -1
	}
	return ap
}

func findAP(g *graph.Graph, i, d int) {
	visited[i] = true
	depth[i] = d
	low[i] = d
	childCount := 0
	isAP := false
	min := func(l, ladj int) int {
		if l < ladj {
			return l
		}
		return ladj
	}
	for _, adj := range g.Nodes[i].Edges {
		if !visited[adj] {
			parent[adj] = i
			findAP(g, adj, d+1)
			childCount = childCount + 1
			if low[adj] >= depth[i] {
				isAP = true
			}
			low[i] = min(low[i], low[adj])
		} else if adj != parent[i] {
			low[i] = min(low[i], depth[adj])
		}
	}
	if (parent[i] >= 0 && isAP) || (parent[i] == -1 && childCount > 1) {
		ap = append(ap, i)
	}
}
