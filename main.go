package main

import "os"

func main() {
	TestMaxEntropy()
	os.Exit(0)
	TestPerceptron()
	TestKdTree()
	TestBayes()
	TestID3()
	TestC45()
}
