package main

import "fmt"

type Perceptron struct {
	w            []float64
	b            float64
	maxIteration int     //最大迭代次数
	learningRate float64 //学习率
	dim          int
}

//感知机的训练过程
func (p *Perceptron) Train(samples [][]float64, labels []int) *Perceptron {
	p.w = make([]float64, p.dim)
	p.b = 0.0 //系数初始化

	for i := 0; i < p.maxIteration; i++ {
		fmt.Printf("%d th Iteration. w=%v b=%v\n", i, p.w, p.b)
		//第i次调整
		trained := true
		//针对每一个不正确的分类进行调整
		for index, sample := range samples {
			//判断是否已经正确分类
			pre := p.Predict(sample)
			if pre != labels[index] {
				//调整系数
				for dim := 0; dim < p.dim; dim++ {
					p.w[dim] += p.learningRate * float64(labels[index]) * samples[index][dim]
					p.b += p.learningRate * float64(labels[index])
				}
				trained = false
			}
		}
		if trained {
			break
		}
	}

	return p
}

//感知机预测过程
func (p *Perceptron) Predict(d []float64) int {
	if len(d) != p.dim {
		fmt.Printf("error input dims %d, expect %d", len(d), p.dim)
		return -1 //输入数据维度不对，可以使用error
	}
	r := 0.0
	for k, v := range d {
		r += p.w[k] * v
	}
	r += p.b
	if r > 0.0 {
		return 1
	}
	return -1
}

func CreateEm(samples [][]float64, labels []int, maxIteration int, learningRate float64) *Perceptron {
	if len(samples) != len(labels) || len(samples) <= 0 {
		return nil
	}
	return (&Perceptron{maxIteration: maxIteration, dim: len(samples[0]), learningRate: learningRate}).Train(samples, labels)
}

func TestEm() {
	fmt.Println("--------- EM Test---------------------")
	set := [][]float64{{3, 3}, {4, 3}, {1, 1}}
	labels := []int{1, 1, -1}
	p := CreateEm(set, labels, 10000, 0.001)
	for i := 0; i < len(set); i++ {
		fmt.Printf("(%f,%f) predict result=%d, expect %d\n", set[i][0], set[i][1], p.Predict(set[i]), labels[i])
	}
	fmt.Printf("(1.5,2.5) predict result=%d.\n", p.Predict([]float64{1.5, 2.5}))
}
