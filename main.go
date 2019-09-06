package main

import (
	"github.com/pkg/profile"
)

func main() {
	defer profile.Start().Stop()
	TestPerceptron()
	TestKdTree()
	TestBayes()
	TestID3()
	TestC45()
	TestMaxEntropy()
	TestSVM()
}
