package defaultmap

import (
	"sync"
)

// Thread safe map
type DefaultSafemap[K comparable, V any] interface {
	Get(key K) V
	Set(key K, val V)
	Count() int
}

type defaultmapImpl[K comparable, V any] struct {
	data        map[K]V
	mutex       sync.RWMutex
	defaultFunc func() V
}

// Thread safe map
func New[K comparable, V any](defaultFunc func() V) DefaultSafemap[K, V] {
	return &defaultmapImpl[K, V]{
		data:        make(map[K]V),
		mutex:       sync.RWMutex{},
		defaultFunc: defaultFunc,
	}
}

func (h *defaultmapImpl[K, V]) Get(key K) V {
	h.mutex.RLock()
	v, ex := h.data[key]
	if !ex {
		v = h.defaultFunc()
		h.data[key] = v
	}
	h.mutex.RUnlock()
	return v
}

func (h *defaultmapImpl[K, V]) Set(key K, val V) {
	h.mutex.Lock()
	h.data[key] = val
	h.mutex.Unlock()
}

func (h *defaultmapImpl[K, V]) Count() int {
	return len(h.data)
}
