# Codeforces Round #747

## [1594A - Consecutive Sum Riddle](https://codeforces.com/contest/1594/problem/A)

属实脑筋急转弯，$l=-n+1$，$r=n$，直接出结果。各路奇葩方法只要合理都能过。

## [1594B - Special Numbers](https://codeforces.com/contest/1594/problem/B)

这个涉及进制知识，我们来复习一下$R$进制。

首先注意组成的 Special Numbers 中第$k$小的数，因为$1\le k \le 10^9$，不可能枚举
，故找规律。规律也很好找，把$k$写作二进制，设从低到高
第$i$为$d_i\quad (i=0,1, 2, \ldots)$，则第$k$小的数正
是$\sum_{i=0}^{\infty}n^i[d_i]$。求此数即可。

## [1594C - Make Them Equal](https://codeforces.com/contest/1594/problem/C)

又没做出 C 题。

看到 Editorial 的 Hint 时我像被雷劈了一样……

![image-20211009214640265](media/Untitled/image-20211009214640265.png)

没错没错，这非常地 CodeForces。

首先我们发现一个绝招：假设我们选了两个数$x_1$和$x_2$，$\gcd(x_1, x_2)=1$，那么所
有的字符都将被覆盖，也就是最多只需要两次操作。

零次操作也很简单，$s$中全部字符等于$c$时。

一次操作想想也明白，只要（开始不说人话）

$$
\exists x, 使得s[i]=c \quad \text{with}\quad i|x
$$
