package main

import (
	"fmt"
)

type Bayes struct {
	Samples [][]string
	Labels  []int
	YProb   []float64                            //Y的先验概率
	Lamda   float64                              //防0
	XProb   map[int](map[int]map[string]float64) //条件概率,多层的map，有点傻
}

func CreateBayes(samples [][]string, labels []int, lamda float64) *Bayes {
	if len(samples) != len(labels) || len(samples) <= 0 {
		fmt.Println("error input data.")
		return nil
	}
	return (&Bayes{Samples: samples, Labels: labels}).Train()
}

func (b *Bayes) Train() *Bayes {
	//首先，确定Y的先验概率
	posCount := 0.0
	for _, v := range b.Labels {
		if v == 1 {
			posCount += 1.0
		}
	}
	b.YProb = make([]float64, 2)
	b.YProb[0] = (posCount + b.Lamda) / (float64((len(b.Labels))) + 2*b.Lamda)
	b.YProb[1] = 1.0 - b.YProb[0]
	fmt.Println(b.YProb)
	//统计各个维度分布
	b.XProb = make(map[int](map[int]map[string]float64))
	b.XProb[1] = make(map[int]map[string]float64)
	b.XProb[0] = make(map[int]map[string]float64)
	for index, sample := range b.Samples {
		for key, value := range sample {
			if b.XProb[b.Labels[index]] == nil {
				b.XProb[b.Labels[index]] = make(map[int]map[string]float64)
			}
			if b.XProb[b.Labels[index]][key] == nil {
				b.XProb[b.Labels[index]][key] = make(map[string]float64)
			}
			b.XProb[b.Labels[index]][key][value] += 1.0
		}
	}
	return b
}

//计算各个维度的条件概率
func (b *Bayes) Prob(dim int, value string, label int) float64 {
	count := float64(b.XProb[label][dim][value]) + b.Lamda
	total := 0.0
	for _, v := range b.XProb[label][dim] {
		total += float64(v) + b.Lamda
	}
	return count / total
}

//预测过程
func (b *Bayes) Predict(d []string) (int, float64) {
	if len(b.Samples) == 0 || len(d) != len(b.Samples[0]) {
		fmt.Println("model not trained.")
		return -1, 0.0
	}
	p0, p1 := 1.0, 1.0
	for dim, value := range d {
		p0 = p0 * b.Prob(dim,value,1)
		p1 = p1 * b.Prob(dim,value,-1)
	}
	if p0*b.YProb[0] > p1*b.YProb[1] {
		return 1, p0 * b.YProb[0]
	}
	return -1, p1 * b.YProb[1]
}


func TestBayes() {
	fmt.Println("--------- Bayes Test---------------------")
	set := [][]string{{"1", "S"}, {"1", "M"}, {"1", "M"}, {"1", "S"}, {"1", "S"}, {"2", "S"}, {"2", "M"}, {"2", "M"}, {"2", "L"},
		{"2", "L"}, {"3", "L"}, {"3", "M"}, {"3", "M"}, {"3", "L"}, {"3", "L"}}
	labels := []int{-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1}
	p := CreateBayes(set, labels, 1.0)
	for i := 0; i < len(set); i++ {
		res, prob := p.Predict(set[i])
		fmt.Printf("(%s,%s) predict result=%d with possibility=%f, expect %d\n", set[i][0], set[i][1], res, prob, labels[i])
	}
	res, prob := p.Predict([]string{"2", "S"})
	fmt.Printf("(2,S) predict result=%d with prob=%f.\n", res, prob)
}
