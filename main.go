package main

import (
	"fmt"
	"math"
	"sort"
)

type Node struct {
	Data                          []int //当前保存的数据
	Dim                           int   //当前点使用的切分维度
	Parent, LeftChild, RightChild *Node //左右子树
}

func (n *Node) SetLeftChild(l *Node) {
	n.LeftChild = l
}
func (n *Node) SetRightChild(r *Node) {
	n.RightChild = r
}

//递归的建立子树
func BuildKdTree(trainSet [][]int, depth int, parent *Node) *Node {
	SampleCount := len(trainSet)
	if SampleCount <= 0 {
		return nil
	}
	SampleDims := len(trainSet[0])
	if SampleDims <= 0 {
		return nil
	}
	//首先，确定本次切分的维度
	cutDim := depth % SampleDims
	//将输入的样本点按照cutDim指定的维度进行排序
	sort.Slice(trainSet, func(i, j int) bool {
		return trainSet[i][cutDim] < trainSet[j][cutDim]
	})
	//按照排序后的结果进行迭代建树,中位数与常规概念不一样，直接除2
	root := &Node{Data: trainSet[SampleCount/2], Parent: parent, Dim: depth % SampleDims}
	//迭代建立左子树和右子树
	if len(trainSet) >= SampleCount/2 {
		root.SetLeftChild(BuildKdTree(trainSet[:SampleCount/2], depth+1, root))
		root.SetRightChild(BuildKdTree(trainSet[SampleCount/2+1:], depth+1, root))
	}
	return root
}

//查找最近的点
func (t *Node) FindNearestPointAndDistance(point []int) (*Node, float64) {
	node := &Node{Data: point}
	destNode := t.findSmallestSubSpace(node, t)
	return t.searchUp(node, destNode, destNode, t.distance(destNode, node))
}

//计算两个节点之间的距离
func (t *Node) distance(a, b *Node) float64 {
	distance := math.MaxFloat64
	if a == nil || b == nil || len(a.Data) != len(b.Data) || len(a.Data) == 0 {
		return distance
	}
	distance = 0.0
	for k, v := range a.Data {
		distance += float64((float32(v) - float32(b.Data[k])) * (float32(v) - float32(b.Data[k])))
	}
	return math.Sqrt(distance)
}

//向上回溯搜索
func (t *Node) searchUp(point *Node, node *Node, currentNearestPoint *Node, currentNearestDistance float64) (*Node, float64) {
	if node.Parent == nil {
		return currentNearestPoint, currentNearestDistance
	}
	//fmt.Printf("in searchup ,currenctNearestDistance=%f\n",currentNearestDistance)

	//与父节点比较距离
	distance := t.distance(node.Parent, point)

	if distance < currentNearestDistance {
		currentNearestDistance = distance
		currentNearestPoint = node.Parent
	}

	//比较与右边子树的距离
	distance = t.distance(point, node.Parent.LeftChild)
	if distance < currentNearestDistance {
		p, d := t.searchDown(point, node.Parent.LeftChild)
		if d < currentNearestDistance {
			currentNearestDistance = d
			currentNearestPoint = p
		}
	}

	//继续迭代向上搜索是否有更近的
	return t.searchUp(point, node.Parent, currentNearestPoint, currentNearestDistance)
}

//向下搜索的话，就是全部遍历
func (t *Node) searchDown(point *Node, node *Node) (*Node, float64) {
	nearestDistance := t.distance(point, node)
	nearestPoint := node

	if node.LeftChild != nil {
		p, d := t.searchDown(point, node.LeftChild)
		if d < nearestDistance {
			nearestDistance = d
			nearestPoint = p
		}
	}
	if node.RightChild != nil {
		p, d := t.searchDown(point, node.RightChild)
		if d < nearestDistance {
			nearestDistance = d
			nearestPoint = p
		}
	}

	return nearestPoint, nearestDistance
}

//找到point点所属的最小区间
func (t *Node) findSmallestSubSpace(point *Node, node *Node) *Node {
	if point.Data[node.Dim] < node.Data[node.Dim] {
		if node.LeftChild == nil {
			return node
		} else {
			return t.findSmallestSubSpace(point, node.LeftChild)
		}
	} else {
		if node.RightChild == nil {
			return node
		} else {
			return t.findSmallestSubSpace(point, node.RightChild)
		}
	}
	return nil
}

func main() {
	trainSet := [][]int{{2, 3, 1}, {5, 4, 2}, {9, 6, 3}, {4, 7, 4}, {8, 1, 5}, {7, 2, 6}}
	destination := []int{3, 5, 1}
	kdTree := BuildKdTree(trainSet, 0, nil)
	p, d := kdTree.FindNearestPointAndDistance(destination)
	fmt.Printf("(3,5,1) nearest neighbor is (%d,%d,%d) with distance=%f.", p.Data[0], p.Data[1],p.Data[2], d)
}
