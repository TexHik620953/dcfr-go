package linq

import "math"

type Number interface {
	float32 | float64 | int32 | int64
}

func Select[T any, K any](data []T, selector func(T) K) []K {
	r := make([]K, len(data))
	for i, dat := range data {
		r[i] = selector(dat)
	}
	return r
}
func Where[T any](data []T, cond func(T) bool) []T {
	r := make([]T, 0, len(data))
	for _, dat := range data {
		if cond(dat) {
			r = append(r, dat)
		}
	}
	return r
}

func Sum[T any](data []T, selector func(T) float64) float64 {
	r := float64(0)
	for _, dat := range data {
		r += selector(dat)
	}
	return r
}

func Mean[T any](data []T, selector func(T) float64) float64 {
	if len(data) == 0 {
		return 0
	}
	return Sum(data, selector) / float64(len(data))
}

func Std[T any](data []T, selector func(T) float64) float64 {
	if len(data) == 0 {
		return 0
	}
	mean := Mean(data, selector)

	sum := float64(0)
	for _, dat := range data {
		temp := selector(dat) - mean
		sum += temp * temp
	}

	return math.Sqrt(sum / float64(len(data)))
}

func Max[T Number](data []T) T {
	if len(data) == 0 {
		return 0
	}
	r := data[0]
	for _, dat := range data {
		if dat > r {
			r = dat
		}
	}
	return r
}
func Min[T Number](data []T) T {
	if len(data) == 0 {
		return 0
	}
	r := data[0]
	for _, dat := range data {
		if dat < r {
			r = dat
		}
	}
	return r
}

func ArgMax[T any](data []T, selector func(T) float64) T {
	var result T
	if len(data) == 0 {
		return result
	}
	result = data[0]
	val := selector(data[0])
	for _, dat := range data {
		v := selector(dat)
		if v > val {
			v = val
			result = dat
		}
	}
	return result
}
func ArgMin[T any](data []T, selector func(T) float64) T {
	val := math.MaxFloat64
	r := data[0]
	for _, dat := range data {
		v := selector(dat)
		if v < val {
			v = val
			r = dat
		}
	}
	return r
}
