package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

//最大熵模型进行分类

//使用mnist数据集

type MnistSample struct {
	Label    int
	Features []int
}

type MaxEntropy struct {
	samples []*MnistSample //样本集合
	test    []*MnistSample
	w       []float64 //最大熵模型的系数
}

//从文件中加载数据
func loadData() []*MnistSample {
	//存储样本集
	var samples []*MnistSample
	//从文件中读取所有样本数据
	fi, err := os.Open("./train.csv")
	if err != nil {
		panic(fmt.Sprintf("read data failed:%v.\n", err))
	}
	defer fi.Close()

	//这个的按照行读取数据有一个大坑，有缓冲区长度限制，默认4096，需要调整
	//br := bufio.NewReader(fi)
	br := bufio.NewReaderSize(fi, 4096*10)

	//忽略第一行
	br.ReadLine()

	for {
		a, _, c := br.ReadLine()
		if c == io.EOF {
			break
		}
		ar := strings.Split(string(a), ",")
		arc := make([]int, len(ar))
		for k, v := range ar {
			arc[k], _ = strconv.Atoi(v)
		}

		sample := &MnistSample{
			Label:    arc[0],
			Features: arc[1:],
		}
		samples = append(samples, sample)
	}

	return samples

}

//创建模型
func CreateMaxEntropyModel() *MaxEntropy {
	d := loadData()
	Shuffle(d) //读取和打乱输入的样本数据
	return (&MaxEntropy{samples: d[:len(d)-5000], test: d[len(d)-5000:]}).train()
}

//模型训练
func (m *MaxEntropy) train() *MaxEntropy {
	if len(m.samples) <= 0 {
		panic("no train data provided.")
	}
	//分配和初始化全0的系数
	m.w = make([]float64, len(m.samples[0].Features))
	//还有特征函数Fi(x,y)需要给定
	//特征函数表示输入x和输出y之间是否满足某一事实
	//一般取指示函数，也可以是任意实值函数

	return m
}

//模型预测
func (m *MaxEntropy) Predict(feature []int) int {
	//预测的过程就是在w给定情况下，Pw(y,x)取得最大值的y
	return 0
}

//最大熵模型的测试
func TestMaxEntropy() {
	m := CreateMaxEntropyModel()
	acc := 0
	for _, v := range m.test {
		if m.Predict(v.Features) == v.Label {
			acc += 1
		}
	}
	fmt.Printf("accuracy:=%f\n", float64(acc)/float64(len(m.test)))
}
