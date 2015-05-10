---
layout: post
title:  "#04 Lagrangeの未定乗数法"
date:   2015-05-11 21:11
categories: ml
---

1変数関数の制約条件付き最大値・最小値を求める際は，導関数の零点から極値をとる値を求め，実際に極値同士の値を比較して最大値・最小値を求めていた．

多変数関数の場合は，

<div>
\[
	\frac{\partial f(\boldsymbol{a})}{\partial x_1} = 0,\cdots,\frac{\partial f(\boldsymbol{a})}{\partial x_n} = 0
\]
</div>

だからといって\\(f(\boldsymbol{a})\\)が極値になっているとは限らない．

例えば\\(f(x,y)=x^2+y^3\\)のグラフをプロットすると次のようになる．

<img src="{{ site.baseurl }}/images/04/critical_point.png" alt="臨界点" style="width: 60%" />

この関数の\\((x,y)=(0,0)\\)では，\\(x,y\\)ともに偏導関数は\\(0\\)になっているが，極値にはなっていない．
（極値であるかどうかに関わらず，偏導関数がすべて\\(0\\)になるような点を停留点，または臨界点という）

1変数の場合でも極値にならない停留点は存在する（例えば\\(f(x)=x^3\\)の\\(x=0\\)）が，1変数関数の場合は増減表を書いて確かめることができる．
しかし多変数関数では増減表が書けない．ここに制約条件が付いてくると更に複雑になる．

このような多変数関数の制約条件付き最大最小問題を解くにはLagrangeの未定乗数法が用いられる．

## Lagrangeの未定乗数法

制約条件\\(g(\boldsymbol{x})=0\\)の下で多変数関数\\(f(x)\\)の極値を与える\\(\boldsymbol{x}\\)は，次のような関数（Lagrange関数という）

<div>
\[
	\tilde{f}(\boldsymbol{x},\lambda) = f(\boldsymbol{x})+\lambda g(\boldsymbol{x})
\]
</div>

に対して，以下の連立方程式の解として与えられる．

<div>
\begin{align}
	\frac{\partial \tilde{f}}{\partial\boldsymbol{x}} &= 0 \\
	\frac{\partial \tilde{f}}{\partial\lambda} &= 0
\end{align}
</div>

<hr />

#### 例1 maximize \\(f(x,y)=2x+3y\\) s.t. \\(x^2+y^2=1\\)

次のようなLagrange関数を作る．

<div>
\[
	\tilde{f}(x,y,\lambda)=2x+3y+\lambda(x^2+y^2-1)
\]
</div>

続いて\\(\tilde{f}\\)を\\(x,y,\lambda\\)でそれぞれ偏微分して「=0」とする．

<div>
\begin{align}
	\frac{\partial\tilde{f}}{\partial x} &= 2+2\lambda x = 0 \\
	\frac{\partial\tilde{f}}{\partial y} &= 3+2\lambda y = 0 \\
	\frac{\partial\tilde{f}}{\partial\lambda} &= x^2+y^2-1 = 0 \\
\end{align}
</div>

この方程式から\\(\lambda\\)を消去すると

<div>
\[
	(x,y) = (\pm\frac{2}{\sqrt{13}},\pm\frac{3}{\sqrt{13}})
\]
</div>

という解が得られる．これらの組は最大値を与える変数の候補になっている．
実際に\\(f(x,y)\\)に代入してみると，\\((x,y)=(\frac{2}{\sqrt{13}},\frac{3}{\sqrt{13}}\\)のときに最大値\\(\sqrt{13}\\)をとり，
\\((x,y)=(-\frac{2}{\sqrt{13}},-\frac{3}{\sqrt{13}}\\)のときに最小値\\(\sqrt{13}\\)をとることがわかる．

実際にグラフをプロットしてみると以下のようになる．

<img src="{{ site.baseurl }}/images/04/lagrange.png" alt="未定乗数法の例" style="width: 60%" />

#### 練習1 例1のプロットをPythonを使って作成してみよ．

[\#03]({{ site.baseurl }}/ml/2015/05/10/multivariate-analysis.html)や[matplotlibのチュートリアル](http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html)を参考にするとよい．
3D空間で線を描画するには``Axes3D.plot``，点(散布図)を描画するには``Axes3D.scatter``を用いる．

### Lagrangeの未定乗数法の原理

厳密な証明は難しく，この分科会の内容から少々逸脱してしまうので，ここでは割愛する．
イメージを掴みたい人は，[昨年の分科会の資料](https://github.com/levelfour/machine-learning-2014/wiki/%E7%AC%AC4%E5%9B%9E---Lagrange%E3%81%AE%E6%9C%AA%E5%AE%9A%E4%B9%97%E6%95%B0%E6%B3%95#%E8%A8%BC%E6%98%8E)を参考にするとよい．

## 最急降下法
