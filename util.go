package main

import (
	"math/rand"
	"reflect"
	"time"
)

// Shuffle reorders in-place the values of an slice
func Shuffle(slice interface{}) {
	swap := reflect.Swapper(slice)
	random := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := reflect.ValueOf(slice).Len() - 1; i > 0; i-- {
		j := random.Intn(i + 1)
		swap(i, j)
	}

}
