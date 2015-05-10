---
layout: post
title:  "#03 多変数解析"
date:   2015-05-10 00:47
categories: ml
---

## 多変数関数

1変数関数は\\(x\\)に対して\\(y\\)を一意に対応付ける関係のことであった．この関係を\\(y=f(x)\\)等と書いた．

多変数関数は，\\(n\\)個の変数\\(x\_1,x\_2,\cdots,x\_n\\)を\\(y\\)に対応づける関係のことであり，
これを\\(y=f(x\_1,x\_2,\cdots,x\_n)\\)，または\\(y=f(\boldsymbol{x})\\)と書いたりする．

<div>
\[
	\boldsymbol{x} = (x_1 x_2 \cdots x_n)^{\mathrm{T}} \longmapsto f(\boldsymbol{x})
\]
</div>

というようにベクトル\\(\boldsymbol{x}\\)と\\(f(\boldsymbol{x})\\)の対応付けとして見ることもできる．
こちらの見方の方が後々微分等で直感的に理解しやすいという利点がある．

#### 例1
多変数関数

<div>
\[
	f(\boldsymbol{x}) = \exp\left(-\frac{(x-0.5)^2+(y-0.2)^2}{2}\right)
\]
</div>

をプロットする．``matplotlib``を用いると，以下のようなコードでプロットできる．

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y = np.meshgrid(np.arange(-3,3,0.1), np.arange(-3,3,0.1))
z = np.exp(-((x-0.5)**2+(y-0.2)**2)/2)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(x,y,z)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

<img src="{{ site.baseurl }}/images/03/gaussian.png" alt="例1" style="width: 60%" />

## 偏微分

1変数関数と同様に多変数関数でも微分を行うが，多変数関数では引数が複数個あるので，どの変数で微分するかで導関数が変化する．

関数\\(f(x\_1,\cdots,x\_n)\\)の引数のうち\\(x\_i\\)以外を固定（定数と見なす）し，\\(x\_i\\)で微分したときの導関数を

<div>
\[
	\frac{\partial f}{\partial x_i}
\]
</div>

と書き，偏導関数という．

## ベクトルによる微分

多変量解析では

<div>
\[
	\frac{\partial(\boldsymbol{a}^{\mathrm{T}}\boldsymbol{x})}{\partial\boldsymbol{x}}
\]
</div>

のようにベクトルで微分を行う操作が式変形の過程で登場することがある．
これは以下の略記である．

<div>
\begin{align}	
	\frac{\partial(\boldsymbol{a}^{\mathrm{T}}\boldsymbol{x})}{\partial\boldsymbol{x}}
	&= \frac{\partial(a_1x_1+\cdots+a_nx_n)}{\partial \left(\begin{array}{c} x_1 \\ : \\ x_n \end{array}\right)} \\
	&= \left(\begin{array}{c} \frac{\partial}{\partial x_1}(a_1x_1+\cdots+a_nx_n) \\ : \\ \frac{\partial}{\partial x_n}(a_1x_1+\cdots+a_nx_n) \end{array}\right) \\
	&= \left(\begin{array}{c} a_1 \\ : \\ a_n \end{array}\right) \\
	&= \boldsymbol{a}
\end{align}
</div>

このように，ベクトルで微分する場合も結果はスカラーの場合から連想される直感的な結果になる．

2次形式

<div>
\[
	\boldsymbol{x}^{\mathrm{T}}\boldsymbol{Bx}=\sum_{i=1}^n\sum_{j=1}^n b_{ij}x_ix_j
\]
</div>

の微分も頻出なので，結果のみ示しておく．興味のある人は手を動かして計算してみるとよい．

<div>
\begin{align}
	\frac{\partial\boldsymbol{x}^{\mathrm{T}}\boldsymbol{Bx}}{\partial\boldsymbol{x}}
	&= (\boldsymbol{B}+\boldsymbol{B}^{\mathrm{T}})\boldsymbol{x} \\
	&= 2\boldsymbol{Bx}^{\mathrm{T}} \; (\boldsymbol{B}:\mathrm{symmetric\,matrix})
\end{align}
</div>
