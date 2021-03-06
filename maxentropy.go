package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"
)

//最大熵模型进行分类

//使用mnist数据集



type MaxEntropy struct {
	sync.Mutex
	samples []*MnistSample //样本集合
	test    []*MnistSample
	w       map[int]float64    //最大熵模型的系数
	Px      map[string]float64 //x出现的频率
	Pxy     map[int]float64    //y出现的频率
	EPxyf   map[int]float64    //特征函数关于经验分布的期望值
	EPxf    map[int]float64    //特征函数关于模型和X经验分布的期望值
	N       float64            //样本容量的大小的倒数
	n       int                //（x,y）的对数 pair，不是log
	M       float64            //类似于学习率
	cvt     map[string]int     //（x,y）对和index之间的转换
	fxyv    map[string]float64 //fxy的函数值
}


//创建模型
//训练费时间，注意改训练次数
func CreateMaxEntropyModel() *MaxEntropy {
	d := loadData()
	Shuffle(d) //读取和打乱输入的样本数据
	return (&MaxEntropy{samples: d[:len(d)-10000], test: d[len(d)-10000:]}).makeIndex()
}

//创建坐标转换数据
func (m *MaxEntropy) makeIndex() *MaxEntropy {
	m.fxyv = make(map[string]float64, 10*len(m.samples[0].Features)*2)
	for i := 0; i <= 9; i++ {
		for j := 0; j < len(m.samples[0].Features); j++ {
			m.fxyv[fmt.Sprintf("%d_%d:%d", j, 0, i)] = 0.0
			m.fxyv[fmt.Sprintf("%d_%d:%d", j, 1, i)] = 0.0
		}
	}
	id := 0
	m.cvt = make(map[string]int)
	for _, v := range m.samples {
		for _, dimValue := range v.Features {
			/*
				if _, ok := m.cvt[fmt.Sprintf("%s:%d", dimValue, v.Label)]; !ok {
					m.cvt[fmt.Sprintf("%s:%d", dimValue, v.Label)] = id
					id++
				}
			*/
			if _, ok := m.cvt[dimValue+convert[v.Label]]; !ok {
				m.cvt[dimValue+convert[v.Label]] = id
				id++
				//这个用于实现fxy函数，就是判断(x,y)对是否存在
				m.fxyv[dimValue+convert[v.Label]] = 1.0
			}
		}
	}

	return m
}

var convert = []string{":0", ":1", ":2", ":3", ":4", ":5", ":6", ":7", ":8", ":9"}

//下标转换
func (m *MaxEntropy) xy2id(x string, y int) int {
	//这个调用次数非常多，想办法加速
	//return m.cvt[fmt.Sprintf("%s:%d", x, y)]
	//每个loop调用的时间，由140秒左右降低到40秒左右
	return m.cvt[x+convert[y]] //相比fmt.Sprintf()的方式，速度提升60%以上
	//字符串拼接看是否有更快的方式
}

func (m *MaxEntropy) id2xy(id int) (string, int) {
	for key, v := range m.cvt {
		if v == id {
			ar := strings.Split(key, ":")
			y, _ := strconv.Atoi(ar[1])
			return ar[0], y
		}
	}
	return "", 0
}

//计算各维度特征出现的概率和联合概率
func (m *MaxEntropy) calcPxyPx() *MaxEntropy {
	m.Px, m.Pxy = make(map[string]float64), make(map[int]float64)
	for _, v := range m.samples {
		for _, feature := range v.Features {
			m.Pxy[m.xy2id(feature, v.Label)] += 1
			m.Px[feature] += 1
		}
	}
	return m
}

//计算特征函数关于经验分布的期望值
func (m *MaxEntropy) calcEpxyf() *MaxEntropy {
	if m.EPxyf == nil {
		m.EPxyf = make(map[int]float64, m.n)
	}
	for i := 0; i < m.n; i++ {
		m.EPxyf[i] = (m.Pxy[i]) * m.N
	}
	return m
}

//计算特征函数关于现有模型和经验分布的期望值
func (m *MaxEntropy) calcEpxf() *MaxEntropy {
	if m.EPxf == nil {
		m.EPxf = make(map[int]float64, m.n)
	}
	//这里是重新分配一段存储更快，还是逐个清除，待比较
	//不过不要太沉迷微观上的优化
	for k, _ := range m.EPxf {
		m.EPxf[k] = 0.0
	}
	var wg sync.WaitGroup
	for _, sample := range m.samples {
		wg.Add(1)
		go func(sample *MnistSample) {
			defer wg.Done()
			pyxes := m.calcProb(sample.Features)
			m.Lock()
			for _, x := range sample.Features {
				for _, pyx := range pyxes {
					//if m.fxy(x, pyx.y) {
					m.EPxf[m.xy2id(x, pyx.y)] += pyx.p * m.N * m.fxy(x, pyx.y)
					//}
				}
			}
			m.Unlock()
		}(sample)
	}
	wg.Wait()
	return m
}

//指定的f(x,y)函数计算过程,就是(x,y)有没有出现过
func (m *MaxEntropy) fxy(x string, y int) float64 {
	return m.fxyv[x+convert[y]]
	/*
		v, _ := m.EPxyf[m.xy2id(x, y)]
		if v != 0.0 {
			return 1.0
		}
		return 0.0
	*/
}

type TPxy struct {
	p float64
	y int
}

//计算(x,y)出现的概率，预测的过程就是找(x,y)最大的那个y
func (m *MaxEntropy) calcProb(features []string) []*TPxy {

	p := make([]float64, 10)
	totalP := float64(0.0) //做分母的概率总和
	for y := 0; y <= 9; y++ {
		//这里是写死的代码，因为我们已经知道标签是0到9
		p[y] = m.pxy(features, y).p
		totalP += p[y]
	}
	/* 去掉一次没必要的循环
	totalP := float64(0.0) //做分母的概率总和
	for _, v := range p {
		totalP += v
	}*/

	//计算这个feature对应的每个y的概率并且返回
	r := make([]*TPxy, 10)
	for i := 0; i <= 9; i++ {
		r[i] = &TPxy{p: p[i] / totalP, y: i} //去掉appened方式的函数调用
	}
	return r
}

//计算针对特定维度特征计算(x,y)概率
func (m *MaxEntropy) pxy(features []string, y int) *TPxy {
	r := float64(0.0)
	//调用频率太高，考虑继续分拆加速
	for _, x := range features {
		//if m.fxy(x, y) {
		r += m.w[m.xy2id(x, y)] * m.fxy(x, y)
		//}
		//if 分支转成乘法
	}
	//考虑exp函数是否可以加速
	return &TPxy{float64(math.Exp(float64(r))), y}
}

//模型训练
func (m *MaxEntropy) train(maxIteration int) *MaxEntropy {
	if len(m.samples) <= 0 {
		panic("no train data provided.")
	}

	//还有特征函数Fi(x,y)需要给定
	//特征函数表示输入x和输出y之间是否满足某一事实
	//一般取指示函数，也可以是任意实值函数
	m.calcPxyPx()

	//设置变量值
	//m.N用到的都是除法，弄成倒数，换成乘法
	m.N, m.n, m.M = 1.0/float64(len(m.samples)), len(m.Pxy), 1.0/50000.0
	//计算EPxyf
	m.calcEpxyf()
	//迭代优化w
	//分配和初始化全0的系数
	m.w = make(map[int]float64, m.n) //其他系数在(x,y)不存在时f(x,y)=0，所以没有必要存在
	startTs := time.Now().Unix()
	for iter := 0; iter < maxIteration; iter++ {

		//计算特征函数关于分布的期望
		//耗掉了大部分时间，是优化的主要位置
		m.calcEpxf()
		for i := 0; i < m.n; i++ {
			//m.w[i] += float64(1.0) / m.M * float64(math.Log(float64(m.EPxyf[i]/m.EPxf[i])))
			m.w[i] += m.M * float64(math.Log(float64(m.EPxyf[i]/m.EPxf[i]))) //去掉一次除法运算
		}
		fmt.Printf("%d loop, last loop cost %d s.\n", iter+1, time.Now().Unix()-startTs)
		startTs = time.Now().Unix()
		//每10次迭代测试一次
		if iter%10 == 0 {
			m.Test()
		}
	}
	return m
}

//模型预测
func (m *MaxEntropy) Predict(feature []string) int {
	//预测的过程就是在w给定情况下，Pw(y,x)取得最大值的y
	pxy := m.calcProb(feature)
	Y, P := 0, float64(0.0)
	for _, v := range pxy {
		if v.p >= P {
			P = v.p
			Y = v.y
		}
	}
	return Y
}

//训练过程中进行测试
func (m *MaxEntropy) Test() {
	acc := 0
	for _, v := range m.test {
		p := m.Predict(v.Features)
		//fmt.Printf("predict:%d, ground=%d \n", p, v.Label)
		if p == v.Label {
			acc += 1
		}
	}
	fmt.Printf("accuracy:=%f\n", float64(acc)/float64(len(m.test)))
}

//最大熵模型的测试
func TestMaxEntropy() {
	fmt.Println("----------Test max entropy-------------------")
	m := CreateMaxEntropyModel()
	m.train(1)
}
