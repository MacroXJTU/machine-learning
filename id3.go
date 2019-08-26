package main

import (
	"fmt"
	"github.com/emirpasic/gods/sets/hashset"
	"math"
)

//id3算法生成决策树
type Sample struct {
	Features []int //特征值
	Label    int   //标签
}

type TreeNode struct {
	Dim      int         //本节点使用的分类特征
	DimValue int         //特征对应的值
	Label    int         //本节点对应的标签
	Parent   *TreeNode   //父节点
	Children []*TreeNode //子节点
}

//利用树进行预测
func (t *TreeNode) Predict(sample Sample) int {
	node := t
	if node != nil {
		if node.Children == nil {
			return node.Label //如果是叶子节点，直接返回本节点对应的标签值
		}
		//不是叶子节点，继续分类
		for _, v := range node.Children {
			if v.DimValue == sample.Features[node.Dim] {
				return v.Predict(sample)
			}
		}
	}

	return 0 //最好使用error带回错误信息
}

//检查是否已经都属于同一类，同时获取最大分类的标签
func CheckSampleClasses(samples []*Sample) (int, bool) {
	ClassCount := make(map[int]int)
	for _, v := range samples {
		ClassCount[v.Label] = ClassCount[v.Label] + 1
	}
	maxClass := 0
	maxClassCount := 0
	for key, v := range ClassCount {
		if maxClassCount < v {
			maxClass = key
			maxClassCount = v
		}
	}
	if maxClassCount == len(samples) {
		return maxClass, true
	} else {
		return maxClass, false
	}
}

//计算切分后的信息增益，比较的第一项是一样的，不计算，直接计算切分后最大的
func CalcClassGain(samples []*Sample, dim int) float64 {
	set := hashset.New()
	for _, v := range samples {
		set.Add(v.Label)
	}
	//生成切分后的样本集
	r := SplitSamples(samples, dim)
	gain := 0.0
	for _, v := range r {
		gain += CalcGain(v)
	}
	return gain
}

func CalcGain(samples []*Sample) float64 {
	totalSamples := len(samples)
	labelCounts := make(map[int]int)
	for _, v := range samples {
		labelCounts[v.Label] = labelCounts[v.Label] + 1
	}
	g := 0.0
	for _, v := range labelCounts {
		p := float64(v) / float64(totalSamples)
		g += p * math.Log(p)
	}
	return g

}

//切分样本集
func SplitSamples(samples []*Sample, maxGainDim int) [][]*Sample {
	var r [][]*Sample
	set := hashset.New()
	for _, v := range samples {
		set.Add(v.Features[maxGainDim])
	}
	for _, v := range set.Values() {
		var c []*Sample
		for _, sa := range samples {
			if sa.Features[maxGainDim] == v.(int) {
				c = append(c, sa)
			}
		}
		r = append(r, c)
	}
	return r
}

//构建
func BuildID3Tree(samples []*Sample, dims hashset.Set, parent *TreeNode, dimValue int) *TreeNode {

	if len(samples) <= 0 {
		return nil //已经没有节点了，不需要操作
	}
	//选取特征进行分类
	maxClass, isSame := CheckSampleClasses(samples)

	if isSame || (!isSame && len(dims.Values()) <= 0) {
		//属于同一类的单节点树
		//或者 虽然不属于同一类，但是呢，属性都已经使用过了
		node := &TreeNode{
			Dim:      0,        //本节点使用的分类特征
			DimValue: dimValue, //本节点特征值
			Label:    maxClass, //本节点对应的标签
			Parent:   parent,   //父节点
			Children: nil} //子节点
		return node
	}
	//按照特征的值，分出一些子集合，在各个子集合，迭代建立子树，作为父节点的子节点
	maxGain, maxGainDim := -100000000.0, 0
	for _, v := range dims.Values() {
		//针对每一个可选特征计算信息增益
		gain := CalcClassGain(samples, v.(int))
		if gain > maxGain {
			maxGain = gain
			maxGainDim = v.(int)
		}
	}

	//按照maxGainDim将集合切分成若干个子集合
	sampleArrays := SplitSamples(samples, maxGainDim)
	//fmt.Printf("samples split into %d arrays.",len(sampleArrays))
	dims.Remove(maxGainDim)
	node := &TreeNode{
		Dim:      maxGainDim, //本节点使用的分类特征
		DimValue: dimValue,   //本次分类的特征值
		Label:    maxClass,          //本节点对应的标签,树剪枝的时候用上，直接把子节点清空就行
		Parent:   parent,     //父节点
		Children: nil} //子节点
	//迭代建立
	for _, v := range sampleArrays {
		node.Children = append(node.Children, BuildID3Tree(v, dims, node, v[0].Features[maxGainDim]))
	}
	return node
}

func TestID3() {
	fmt.Println("--------- ID3 Decision Tree Test---------------------")
	Samples := []*Sample{&Sample{Features: []int{1, 0, 0, 1}, Label: 0},
		&Sample{Features: []int{1, 0, 0, 2}, Label: 0}, &Sample{Features: []int{1, 1, 0, 2}, Label: 1}, &Sample{Features: []int{1, 1, 1, 1}, Label: 1},
		&Sample{Features: []int{1, 0, 0, 1}, Label: 0}, &Sample{Features: []int{2, 0, 0, 1}, Label: 0}, &Sample{Features: []int{2, 0, 0, 2}, Label: 0},
		&Sample{Features: []int{2, 1, 1, 2}, Label: 1}, &Sample{Features: []int{2, 0, 1, 3}, Label: 1}, &Sample{Features: []int{2, 0, 1, 3}, Label: 1},
		&Sample{Features: []int{3, 0, 1, 3}, Label: 1}, &Sample{Features: []int{3, 0, 1, 2}, Label: 1}, &Sample{Features: []int{3, 1, 0, 2}, Label: 1},
		&Sample{Features: []int{3, 1, 0, 3}, Label: 1}, &Sample{Features: []int{3, 0, 0, 1}, Label: 0}}

	dims := hashset.New()
	dims.Add(0, 1, 2, 3)
	node := BuildID3Tree(Samples, *dims, nil, 0)

	for _, v := range Samples {
		fmt.Printf("(%d,%d,%d,%d) predicts as %d, expect %d\n", v.Features[0], v.Features[1], v.Features[2], v.Features[3], node.Predict(*v), v.Label)
	}
}
