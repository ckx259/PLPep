# PLPep
基于蛋白质语言模型和深度残差神经网络的蛋白质与肽结合残基预测

识别蛋白质-肽结合残基对于揭示蛋白质功能机制和探索药物发现至关重要。许多计算方法被提出来用于预测肽结合残基。然而，由于特征表示质量的限制，预测性能仍有很大的提升空间。因此，本文设计了一种新的深度学习模型PL-Pep，从蛋白质序列中挖掘出更多判别性信息，从而提高肽结合残基预测性能。具体来说，首先使用预训练的基于深度学习的蛋白质语言模型（ESM2）来提取与蛋白质功能相关的蛋白质序列高潜在判别表示。接着，设计了一个基于卷积的残差神经网络来训练蛋白质-肽结合残基预测模型。此外，改进了对比损失函数以提高模型的性能。两个独立测试数据集的实验结果表明，与大多数现有最先进的预测方法相比，PL-Pep可以获得更高的马修相关系数(MCC)和AUC值。详细的数据分析表明，PL-Pep的主要优势在于利用预训练的蛋白质语言模型，仅从蛋白质序列中提取更多判别性信息以及利用改进的对比损失函数进一步优化不平衡数据集下的结合残基的特征表示。

## 先决条件:
    - Python3.7, Anaconda3
    - Linux system

## 环境准备
* 如果满足前提条件，请忽略以下操作。如果您的Python版本不是3.7，请执行以下操作。

~~~
conda create -n PLPep python==3.7
conda activate PLPep
~~~

## Installation:

* 从https://github.com/ckx259/plpep/ 下载这个存储库。然后，在Linux系统上解压缩并运行以下命令行。

~~~
  $ cd PLPep-master
  $ chmod 777 ./install.sh
  $ ./install.sh
~~~

* 如果包不能在你的计算集群上正常工作，你应该通过运行以下命令来安装依赖项:

~~~
  $ cd PLPep-main
  $ pip install -r requirements.txt
~~~

## 运行案例
* PLPep的预测命令("-m"表示模型名称，"-l"表示CRNN层数，m可以是model_D1tr.pkl或者model_D2tr.pkl,如果选择model_D1tr.pkl则l=3，如果选择model_D2tr.pkl则l=2)
~~~
  $ python predict.py -sf example/results/ -seq_fa example/seq.fa -m model_D1tr.pkl -l 3
~~~


## 结果
* 输入fasta文件(-seq_fa)中每个蛋白质(例如，1dpuA)的预测结果文件(例如，“1dpuA.pred”)可以在您输入为“-sf”的文件夹中找到。

* 每个预测结果文件中有四列。第一列是残基索引。第二列是残基类型。第三列是预测为肽结合残基的概率。第四列为预测结果(“B”和“N”分别表示预测的肽结合残基和非肽结合残基)。例如:
~~~
Index   AA      Prob0[cutoff:0.5]       State
0       A       0.10030283482236421     N
1       N       0.2305866342433313      N
2       G       0.025668953283439103    N
3       L       0.051343766831211377    N
4       T       0.019408047199249268    N
5       V       0.21095703774946703     N
6       A       0.012030509895723967    N
7       Q       0.010916543982148528    N
8       N       0.04713442707139941     N
9       Q       0.026714079281047576    N
10      V       0.014767545931645745    N
11      L       0.0019425360544991655   N
12      N       0.02515005520628796     N
13      L       0.0030931658644448674   N
14      I       0.012935923432873517    N
15      K       0.009504009500413664    N
16      A       0.0013285580018364978   N
17      C       0.013339149008270433    N
18      P       0.0060203934244198384   N
19      R       0.040193256140475184    N
20      P       0.002552176993762308    N
21      E       0.006981122345698049    N
22      G       0.03026768474755928     N
23      L       0.00994720531440694     N
24      N       0.015351806238277973    N
25      F       0.014937674098059893    N
26      Q       0.003710359213556242    N
27      D       0.0068711763844343295   N
28      L       0.005273692034811942    N
29      K       0.006532199705406578    N
30      N       0.00978148318517066     N
31      Q       0.03104891017183626     N
32      L       0.15089101498669474     N
33      K       0.030340731141634047    N
34      H       0.00858085557769247     N
35      M       0.15784288393514656     N
36      S       0.04701913748401697     N
37      V       0.05048890690339903     N
38      S       0.15123227689362398     N
39      S       0.037951549207668904    N
40      I       0.03815351256041918     N
41      K       0.11687966213582436     N
42      Q       0.09049793808168358     N
43      A       0.10428899572341918     N
44      V       0.025812295475631148    N
45      D       0.22482819520193048     N
46      F       0.762390673160553       B
47      L       0.09581810235977173     N
48      S       0.3692213484876946      N
49      N       0.10824843254130112     N
50      E       0.38282002857863817     N
51      G       0.14549830989534587     N
52      H       0.09911035900650268     N
53      I       0.01991896789928501     N
54      Y       0.9331618895069469      B
55      S       0.05582596981914409     N
56      T       0.26169222593307495     N
57      V       0.24301536019123554     N
58      D       0.4003190219497389      N
59      D       0.659671028872071       B
60      D       0.19020813703536987     N
61      H       0.009008245928566264    N
62      F       0.08818975507506151     N
63      K       0.0708825130629165      N
64      S       0.0389248979371999      N
65      T       0.6115981158137883      B
66      D       0.23703180606710622     N
67      A       0.03283380545797469     N
68      E       0.16053220151268835     N
~~~

## 小贴士
* <b>此安装包仅供学术使用</b>. 如果您有任何问题，请发邮件到:junh_cs@126.com

## References
[1] . 基于蛋白质语言模型和深度残差神经网络的蛋白质与肽结合残基预测

