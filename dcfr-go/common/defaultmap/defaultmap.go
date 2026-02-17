package defaultmap

import (
	"sync"
)

// Thread safe map
type DefaultSafemap[K comparable, V any] interface {
	Get(key K) V
	Set(key K, val V)
	Delete(key K)
	Count() int
	Foreach(it func(K, V) bool)
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
	h.mutex.Lock()
	v, ex := h.data[key]
	if !ex {
		v = h.defaultFunc()
		h.data[key] = v
	}
	h.mutex.Unlock()
	return v
}

func (h *defaultmapImpl[K, V]) Set(key K, val V) {
	h.mutex.Lock()
	h.data[key] = val
	h.mutex.Unlock()
}
func (h *defaultmapImpl[K, V]) Delete(key K) {
	h.mutex.Lock()
	delete(h.data, key)
	h.mutex.Unlock()
}

func (h *defaultmapImpl[K, V]) Count() int {
	h.mutex.RLock()
	defer h.mutex.RUnlock()
	return len(h.data)
}

func (h *defaultmapImpl[K, V]) Foreach(it func(K, V) bool) {
	h.mutex.RLock()
	defer h.mutex.RUnlock()
	for k, v := range h.data {
		if !it(k, v) {
			break
		}
	}
}
