---
layout: post
title:  "#02 環境構築"
date:   2015-04-26 21:07
categories: ml
---

## 統計解析とプログラミング

統計処理とプログラミングは切っても切り離せない関係にある。
プログラミングと言っても、この分野ではGUIで華やかなアプリケーションを作る技術やアセンブリをバリバリ読むということは勿論無く、単純に計算したい数式を記述するツールとして用いられることがほとんどだ。
そのため、プログラミング初心者の人も臆する必要はない（むしろプログラミングの観点から見れば難易度は低いので、プログラミングの勉強としても良いはずだ）。

統計解析によく用いられるプログラミング言語・ツールには以下のようなものがある（levelfourのバイアスがかかっている）。

+ Excel: 言わずもがな。<s>方眼紙として使わなければ最強なんだけどなあ</s>
+ MatLab: 数値計算の王道ソフトウェア。ライセンスも高いので研究室向け。
+ R: GNU発のフリーの数値解析ソフトウェア。フリーなのに高機能。
+ Python: 汎用プログラミング言語だが、数値計算ライブラリが充実。
+ (Julia: 2012年に初公開された、LLVMベースの高速数値計算向け汎用プログラミング言語)

この分科会ではPythonを用いることにする。統計解析だけに絞るのであれば確かにR言語の方が（特に検定回りで）ライブラリが充実しているのだが、Pythonの方が汎用プログラミング言語ともあって、他のツールの助力を借りること無くファイルアクセスしたりWeb公開できるのが強みだ。
世界的に見てもこの流れがあるようで、最近はR言語を始めるならPythonをやろうとよく言われる。

## Pythonの統計解析用環境

一般的に以下のようなライブラリ群が用いられる。

+ numpy: 行列計算
+ scipy: 科学計算（フィッティング、連立方程式、特殊関数、微分積分など）
+ matplotlib: グラフプロットツール
+ pandas: データの読み書き、表の作成（Pythonで動くExcelのような感じ）
+ scikit-learn: 機械学習
+ IPython(Jupyter): 高機能インタラクティブPythonシェル

この回では、Python自体の処理系のインストールと、上記ライブラリのインストールを目指す。

## pyenvのインストール

`pyenv`を入れておくとよい。仮想環境を使うと

+ 任意のバージョンのPythonを簡単に入れられる
+ システムのPythonを破壊せずにとっておける
+ Sandbox環境を簡単に作成&破棄できる
+ 後述のAnacondaもあっという間にインストール
+ Ma OS Xだと`brew install pyenv`で入れられるっぽい

といった風にメリットづくしなので、オススメする。ちなみに同じバージョンのPythonを複数入れたいという需要があるかもしれないが、そんなときには`virtualenv`を用いる。

以下OSごとのインストール方法を説明する。levelfourはMac OS 10.9, 10.10, Ubuntu 14.04でのインストールを確認した。
また、2014年度の分科会で他のメンバーによりWindowsでのインストールも確認された。

#### Windows

<s>Cygwinで以下のページの通りに作業するとインストールできることを確認。</s>

<s>[pyenvとvirtualenvのインストールと使い方 - Qiita](http://qiita.com/la_luna_azul/items/3f64016feaad1722805c)</s>

Windowsだと**pyenvからanacondaをうまくインストールできない**ようなので、後述のAnaconda Installerを使用するとよい。

#### Mac OS X

```
$ brew install pyenv
```

#### Linux

[pyenv-installer](https://github.com/yyuu/pyenv-installer)を用いるとインストールできる。
（Ubuntu 14.04で確認）

### pyenvの使い方（概略）

インストール終了後に標準出力に.bashrcに環境変数設定を追記するように促されていると思うので、その通りに従う。

#### インストール可能バージョンを見る

```
$ pyenv install -l
```

#### インストールする

```
$ pyenv install [what-you-want]
```

#### インストール済バージョン一覧を参照

```
$ pyenv versions
```

#### 環境の切り替え

```
$ pyenv global [environment]
```

ちなみに`global`でシステム全体で使用するPythonのバージョン、`local`で**そのディレクトリより下層**で使用するPythonのバージョンを設定することが出来る。

【参考】[pyenvとvirtualenvのインストールと使い方 - Qiita](http://qiita.com/la_luna_azul/items/3f64016feaad1722805c)

## Anacondaのインストール

pyenvをインストールできたところで、次は上で挙げた数値計算ライブラリ群をインストールする。
しかし、これらライブラリのインストールには非常に手間がかかり、ビルドエラーが起こって解決が困難になることも多い。
そこで、Pythonの処理系と有用なライブラリ群をまとめたパッケージである[Anaconda](http://continuum.io/downloads#all)を利用する。
pyenvがインストールされていれば、Anacondaのインストールは非常に簡単である。

```
$ pyenv install anaconda3-2.1.0
```

このインストールには20分〜30分程度の時間を要するので注意すること。
インストールの確認には

```
$ pyenv versions
```

を実行した際にanaconda3-2.1.0が表示されるかを確認すればよい。
各種ライブラリ群が正常動作していることを確認するために、Anaconda環境に切り替える。

```
$ pyenv global anaconda3-2.1.0
```

そしてPythonを立ち上げる。

```
$ python
```

以下の行を一行ずつ入力し、エラーが起こらなければ成功である。

```python
import numpy
import scipy
import matplotlib
import pandas
import sklearn
```

### Windows等pyenvが正常に動作しない環境の場合

[Anacondaのサポートページ](http://continuum.io/downloads#all)からAnaconda installerをダウンロードしてインストールするのが簡単だと思われる。

## Pythonの基礎知識

Pythonは汎用プログラミング言語の一つで、（多くの場合）インタプリタ上で動作する動的型付けスクリプト言語である。
Pythonの大きな特徴としては

+ インデントで関数等のブロックを表現する
+ 柔軟な動的型付け
+ 内包表記による簡潔で高速なコード
+ 関数型プログラミングも可能
+ 豊富な標準ライブラリ、活発な海外コミュニティ

逆にデメリットとしては

+ インデントに縛られる
+ lambda式が書きにくい
+ 正規表現が使いにくい
+ オブジェクト指向にもかかわらずメソッドチェーンが書きにくい

あたりだろうか（個人の感想）。

### バージョン

Pythonは現在2.xから3.xへの移行期を向かえている。3.xでは2.xの後方互換性を切り捨てる形で、新たな機能を取り入れることを目指している。
（Pythonコミュニティとしては「今までよりもPythonicなコード」を目指しているようだ）

そのため、3.xのコードは2.xの処理系では動かないし、その逆もまた然りだ。
昨年度の分科会では2.7の処理系を用いて説明を行ったが、3.xに対する環境も整備しつつあるこの時期が3.xへの移行タイミングと見極め、本分科会では3.xを対象として説明を行う。

ちなみに、上でAnacondaをインストールしてもらった際にanaconda3-2.1.0というパッケージをインストールしてもらった。
これは2015年4月末時点の最新バージョンで、付属しているPython処理系のバージョンは3.4.1である。

### サンプルスクリプト

```python
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 定義域は(-3,3)
    xs = np.arange(-3, 3, 0.1)
    # sinカーブに正規分布ノイズをのせる
    ys = np.array([np.sin(x) + np.random.normal(0,0.1) for x in xs])

    plt.plot(xs, ys, 'o')

    # 3次関数フィッティング
    param = scipy.polyfit(xs, ys, 3, full=True)[0]
    f = scipy.poly1d(param)

    plt.plot(xs, f(xs))
    plt.show()
```

上記のスクリプトを保存してpythonで実行すると、以下のような結果が得られる。
sin関数のフィッティングである。

![sin関数のフィッティング]({{ site.baseurl }}/images/02/sin.png)
