package main

import (
	"bytes"
	"fmt"
	libSvm "github.com/ewalker544/libsvm-go"
	"math"
	"os"
	"sync"
	"sync/atomic"
)

type svmModel struct {
	sync.Mutex
	samples []*MnistSample
	test    []*MnistSample
	param   *libSvm.Parameter //SVM参数
	model   *libSvm.Model     //模型
	file    string            //数据文件名称
}

func CreateSvmModel() *svmModel {
	fmt.Println("loading data from csv file.")
	d := loadData()
	Shuffle(d) //读取和打乱输入的样本数据
	//因为libsvm库使用指定个数输入数据的方式，所以读出的所有数据全部用作测试数据
	fmt.Printf("load data done, %d total items, %d for train and %d for test.\n", len(d), len(d)-10000, 100000)

	return (&svmModel{samples: d[:len(d)-10000], test: d[len(d)-10000:]})
}

func (m *svmModel) Train() *svmModel {
	m.file = fmt.Sprintf("train_svm.csv")
	//首先判断train_svm.csv开头的文件是否存在,存在的话就继续使用
	//这会导致部分训练集的数据出现在测试集中
	//但是先不纠结这些细节，为加速计算
	if _, err := os.Stat(m.file); err != nil && !os.IsExist(err) {
		//输入文件转成libsvm支持的格式
		f, err := os.OpenFile(m.file, os.O_RDWR|os.O_CREATE, 0755)
		if err != nil {
			panic(err)
		}

		for _, v := range m.samples {
			var buffer bytes.Buffer
			buffer.WriteString(fmt.Sprintf("%d", v.Label))
			var total float64
			for _, fv := range v.FeaturesInt {
				total += float64(fv * fv)
			}
			l2 := math.Sqrt(total)
			for index, fv := range v.FeaturesInt {
				if fv == 0 {
					continue
				}
				buffer.WriteString(fmt.Sprintf(" %d:%f", index+1, float64(fv)/l2))
			}

			buffer.WriteString("\n")
			_, _ = f.WriteString(buffer.String())
		}
		_ = f.Close()
	}
	//转换完成，开始训练
	fmt.Println("start training svm model")
	m.param = libSvm.NewParameter()
	//m.param.KernelType = libSvm.POLY //多项式核
	//m.param.SvmType=libSvm.NU_SVC
	m.param.Eps = 1.0e-4 //设置结束条件的精度要求
	m.param.C=3.0
	m.model = libSvm.NewModel(m.param)
	problem, err := libSvm.NewProblem(m.file, m.param)
	if err != nil {
		panic(err)
	}
	_ = m.model.Train(problem)
	fmt.Println("training svm model done")
	return m
}

func (m *svmModel) Test() *svmModel {
	var acc int32
	var wg sync.WaitGroup
	for _, v := range m.test {
		wg.Add(1)
		go func(v *MnistSample) {
			defer wg.Done()
			var total float64
			data := make(map[int]float64, len(m.test[0].FeaturesInt))
			for _, val := range v.FeaturesInt {
				total += float64(val * val)
			}
			l2 := math.Sqrt(total)
			for key, val := range v.FeaturesInt {
				data[key] = float64(val) / l2
			}
			p := m.model.Predict(data)
			if math.Abs(p-float64(v.Label)) < 0.1 {
				atomic.AddInt32(&acc, 1)
			}
			fmt.Printf("INFO:predict as %f, except %d\n", m.model.Predict(data), v.Label)
		}(v)
	}
	wg.Wait()
	fmt.Printf("Accuracy=%f\n", float64(acc)/float64(len(m.test)))
	return m
}

func TestSVM() {
	fmt.Println("----------Test SVM ---------------")
	m := CreateSvmModel().Train().Test()
	_ = m
	//os.Remove(m.file) //清除生成的临时文件
}

//google libsvm输入文件的格式
