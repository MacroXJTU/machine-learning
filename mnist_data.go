package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

type MnistSample struct {
	Label       int
	Features    []string
	FeaturesInt []int
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
	_, _, _ = br.ReadLine()

	for {
		a, _, c := br.ReadLine()
		if c == io.EOF {
			break
		}
		ar := strings.Split(string(a), ",")
		arc := make([]int, len(ar))
		for k, v := range ar[1:] {
			arc[k+1], _ = strconv.Atoi(v)
			//图片二值化
			if arc[k+1] >= 128 {
				arc[k+1] = 1
			} else {
				arc[k+1] = 0
			}
		}
		label, _ := strconv.Atoi(ar[0])
		sample := &MnistSample{
			Label:    label,
			Features: make([]string, len(arc[1:]))}
		//对特征进行处理，不同维度的0和1是不一样的
		for index := 0; index < len(sample.Features); index++ {
			sample.Features[index] = fmt.Sprintf("%d_%d", index, arc[1+index])
			sample.FeaturesInt[index] = arc[1+index]
		}
		samples = append(samples, sample)

	}

	return samples

}
