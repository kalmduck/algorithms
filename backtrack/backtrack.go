package backtrack

/* Backtrack implements a generic backtracking framework to solves problems like
n-queens or finding your way out of a maze. */

// The Problem interface provides a way to describe your problem such that the
// framework understands how to solve it.
type Problem interface {

	// Valid should return true only if p is worth considering
	Valid(p Position) bool

	// Record marks the Position p as tried.  Assumes that p is a valid position.
	Record(p Position)

	// Done checks if the current state of the the problem constitutes a solution.
	Done(p Position) bool

	// Undo backs out of the current position, marking this one as tried but
	// unsuccessful
	Undo(p Position)
}

// Position represents a specific state within the problem set.
type Position interface {

	// NextVal returns the next possible value for the given position in the set.
	NextVal() Position

	// End returns true if we have exhausted all of the possible values at this position
	End() bool

	// NextPos returns the next possible position.
	// if we've reached the end, returns nil
	NextPos() Position
}

type Backtracker struct {
	p Problem
}

// New returns a Backtracker initiallized with the passed Problem set
func New(p Problem) Backtracker {
	return Backtracker{p}
}

func (b Backtracker) Solve(pos Position) bool {
	success := false
	// Loop until we run out of possible values or we find a solution
	for ; !success || pos != nil; pos = pos.NextVal() {
		if b.p.Valid(pos) { // this could be a winner
			b.p.Record(pos)
			if b.p.Done(pos) { // we found a solution!
				success = true
			} else { // we havent sound a solution yet
				success = b.Solve(pos.NextPos()) // continue to the next variable
				if !success {                    // that was a failed path
					b.p.Undo(pos)
				}
			}
		}
	}
	return success
}
