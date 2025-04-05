package bench

import "time"

func MeasureExec(exec func()) time.Duration {
	s := time.Now()
	exec()
	return time.Since(s)
}
