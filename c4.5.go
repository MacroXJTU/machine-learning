package main

import (
	"fmt"
	"github.com/emirpasic/gods/sets/hashset"
)

//计算信息增益比
func CalcClassGainRate(samples []*Sample, dim int) float64 {
	set := hashset.New()
	for _, v := range samples {
		set.Add(v.Label)
	}
	Had := CalcGain(samples) //这是分母
	//生成切分后的样本集
	r := SplitSamples(samples, dim)
	gain := 0.0
	for _, v := range r {
		gain += CalcGain(v) //信息增益的计算，在id3里取反了，原因是被减做差值,但是不影响比值
	}

	return (Had - gain) / Had
}

//构建
func BuildC45Tree(samples []*Sample, dims hashset.Set, parent *TreeNode, dimValue int) *TreeNode {

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
		gain := CalcClassGainRate(samples, v.(int))
		//fmt.Printf("gainRate=%f,current maxGain=%f,maxGainDim=%d\n", gain, maxGain, maxGainDim)
		if gain > maxGain {
			//fmt.Printf("gainRate=%f,current maxGain=%f,maxGainDim=%d\n", gain, maxGain, maxGainDim)
			maxGain = gain
			maxGainDim = v.(int)
		}
	}
	//这里考虑添加maxGain的阈值检查，如果小于阈值，作为一个叶子节点添加

	fmt.Printf("use %d dim as seprate.\n", maxGainDim)
	//按照maxGainDim将集合切分成若干个子集合
	sampleArrays := SplitSamples(samples, maxGainDim)
	//fmt.Printf("samples split into %d arrays.",len(sampleArrays))
	dims.Remove(maxGainDim)
	node := &TreeNode{
		Dim:      maxGainDim, //本节点使用的分类特征
		DimValue: dimValue,   //本次分类的特征值
		Label:    0,          //本节点对应的标签,非叶子节点的Label没有意义
		Parent:   parent,     //父节点
		Children: nil} //子节点
	//迭代建立
	for _, v := range sampleArrays {
		node.Children = append(node.Children, BuildC45Tree(v, dims, node, v[0].Features[maxGainDim]))
	}
	return node
}

func TestC45() {
	fmt.Println("--------- C45 Decision Tree Test---------------------")
	Samples := []*Sample{&Sample{Features: []int{1, 0, 0, 1}, Label: 0},
		&Sample{Features: []int{1, 0, 0, 2}, Label: 0}, &Sample{Features: []int{1, 1, 0, 2}, Label: 1}, &Sample{Features: []int{1, 1, 1, 1}, Label: 1},
		&Sample{Features: []int{1, 0, 0, 1}, Label: 0}, &Sample{Features: []int{2, 0, 0, 1}, Label: 0}, &Sample{Features: []int{2, 0, 0, 2}, Label: 0},
		&Sample{Features: []int{2, 1, 1, 2}, Label: 1}, &Sample{Features: []int{2, 0, 1, 3}, Label: 1}, &Sample{Features: []int{2, 0, 1, 3}, Label: 1},
		&Sample{Features: []int{3, 0, 1, 3}, Label: 1}, &Sample{Features: []int{3, 0, 1, 2}, Label: 1}, &Sample{Features: []int{3, 1, 0, 2}, Label: 1},
		&Sample{Features: []int{3, 1, 0, 3}, Label: 1}, &Sample{Features: []int{3, 0, 0, 1}, Label: 0}}

	dims := hashset.New()
	dims.Add(0, 1, 2, 3)
	node := BuildC45Tree(Samples, *dims, nil, 0)

	for _, v := range Samples {
		fmt.Printf("(%d,%d,%d,%d) predicts as %d, expect %d\n", v.Features[0], v.Features[1], v.Features[2], v.Features[3], node.Predict(*v), v.Label)
	}
}
