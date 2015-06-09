---
layout: post
title:  "#08 確率的勾配降下法"
date:   2015-06-08
categories: ml
---

\#07ではパーセプトロンを紹介した．
ここではパーセプトロンを用いた学習を紹介する．

そもそも教師あり学習における学習とは，\#06で触れたようにデータセット

<div>
\[
	\mathcal{D}=\{(\boldsymbol{x_1},t_1),\cdots,(\boldsymbol{x_N},t_N)\}
\]
</div>

が与えられて，各学習データ\\((\boldsymbol{x\_i},t\_i)\\)に対して\\(z\_i\\)という出力が得られるとき，<u>誤差関数を最小にするようにパラメータを推定すること</u>であった．
また，一般的に誤差関数として二乗誤差関数

<div>
\[
	E(\boldsymbol{w})=\frac{1}{2}\sum_{i=1}^N|t_i-z_i|^2
\]
</div>

が用いられることが多い．(先頭の1/2は微分したときに係数が消えるようにつけてあるだけ)

\#06では単純な線形識別関数を用いて識別していたので，解析的に二乗誤差関数の最小化を行うことができた（正規方程式に落とし込める）．
しかし，パーセプトロンが少し複雑になると解析的に解くのは難しくなるので，\#04で紹介した最適化手法の勾配法を用いて最小化する．

<div>
\[
	\boldsymbol{w}^{(n+1)}=\boldsymbol{w}^{(n)}-\eta\frac{\partial E}{\partial\boldsymbol{w}}
\]
</div>

という更新式でパラメータ\\(\boldsymbol{w}\\)を更新するアルゴリズムであった．

気をつけたいのは，このとき必要な学習データは事前にすべて読み込ませなければいけないという点である．
なぜなら，二乗誤差関数Eを計算するときにはすべての学習データが必要だからである．
このように学習データをまとめて学習する方法は__バッチ学習__と呼ばれている．

ここから紹介するのは，学習データを少しずつ（極端には1つずつ）学習する方法であり，__オンライン学習__と呼ばれる．

## 確率的勾配降下法(SGD)

確率的勾配降下法(Stochastic Gradient Descent = SGD)とは，<u>ランダムに学習データを1つ選んで誤差関数を計算し，その勾配方向にパラメータを修正する操作を反復する</u>手法である．

今，\\(k+1\\)回目のパラメータ更新において\\(n\_{k+1}\\)番目の学習データを選んだとき，更新式は以下のようになる．

<div>
\[
	\boldsymbol{w}^{(k+1)}=\boldsymbol{w}^{(k)}-\eta\frac{\partial}{\partial\boldsymbol{w}}\left(\frac{1}{2}|t_{n_{k+1}}-z_{n_{k+1}}|^2\right) \tag{8-1}
\]
</div>

すべての学習データに対する誤差関数ではなく，\\(n\_{k+1}\\)番目の学習データに対する誤差関数

<div>
\[
	E_{n_{k+1}}(\boldsymbol{w})=\frac{1}{2}|t_{n_{k+1}}-z_{n_{k+1}}|^2 \tag{8-2}
\]
</div>

の勾配方向にパラメータを更新している．

### SGDの利点

SGDにはナイーブな勾配法に比べて以下のような利点があげられる．

+ 局所最適解にトラップしにくい（勾配法の初期値依存問題への解決）
+ 冗長な学習データがある場合，勾配法よりも学習が高速
+ 学習データを収集しながら逐次的に学習できる

## 誤差逆伝播法(Back Propagation)

![多層パーセプトロン]({{ site.baseurl }}/images/08/perceptron.003.jpg)

原理的には(8-1)の更新式を用いれば，SGDにより最適パラメータが推定できる．

図のパーセプトロンに対して(8-2)の微分を計算しようとすると，2段目（青い部分）は

<div>
\begin{align}
	\frac{\partial E_n}{\partial\boldsymbol{v}}
	&= (z_n-t_n)\frac{\partial z_n}{\partial\boldsymbol{v}} \\
	&= (z_n-t_n)\frac{\partial}{\partial\boldsymbol{v}}(\sigma(\boldsymbol{v}^{\mathrm{T}}\boldsymbol{y})) \\
	&= (z_n-t_n)\sigma'\boldsymbol{y}
\end{align}
</div>

1段目は（\\([x\_1, \cdots,x\_4]\mapsto y\_1\\)，すなわち赤い部分にのみ着目すると）

<div>
\begin{align}
	\frac{\partial E_n}{\partial\boldsymbol{w_1}}
	&= (z_n-t_n)\frac{\partial z_n}{\partial\boldsymbol{w_1}} \\
	&= (z_n-t_n)\frac{\partial}{\partial\boldsymbol{w_1}}(\sigma(\boldsymbol{v}^{\mathrm{T}}\boldsymbol{y})) \\
	&= (z_n-t_n)\sigma'v_1\frac{\partial y_1}{\partial\boldsymbol{w_1}} \\
	&= (z_n-t_n)\sigma'v_1\frac{\partial\sigma(\boldsymbol{w_1}^{\mathrm{T}}\boldsymbol{x})}{\partial\boldsymbol{w_1}} \\
	&= (z_n-t_n)\sigma'^2v_1\boldsymbol{x}
\end{align}
</div>

ここで重要なのは，どちらも

<div>
\[
	\frac{\partial(\mathit{Error Function})}{\partial(\overrightarrow{\mathit{Param}})}\propto(\mathit{Error})\times(\overrightarrow{\mathit{Input}})
\]
</div>

という形になっているという点である（\\(z\_n-t\_n\\)が誤差，\\(\boldsymbol{x,y}\\)が入力ベクトル）．

![パーセプトロンの更新]({{ site.baseurl }}/images/08/perceptron.gif)

イメージとしてはこのGIFのように，入力ベクトルの方向にパラメータを修正することを繰り返すことになる
（このGIFでは固定係数\\(\eta\\)による更新だが，ここでは固定係数ではなく誤差がかかった変動係数である）．
