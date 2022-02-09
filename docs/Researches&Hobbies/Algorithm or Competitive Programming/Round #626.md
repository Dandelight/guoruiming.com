## [B. Counting Subrectangles](https://codeforces.com/contest/1323/problem/B)

给定$a_i$和$b_i$两个由$0$和$1$组成的数组，定义矩阵$c_{ij}=a_i\times b_j$，计算矩阵$C$中由$1$组成的面积为$k$的矩形个数。

假设有数组$x, y$满足$x_i \times y_i = k$，可见问题的关键是找到$a$中有多少个连续$x_i$个$1$，$b$中有多少个连续$y_i$个$1$。

于是我们需要使用一个$cnt$数组，$cnt_i$记录$a$中有多少个连续$i$个$1$；$b$数组同理。建立$cnt$数组需扫描$a$数组，如果遇到连续的$\ell$个$1$，就将$cnt_i$增加$\ell - i + 1$。时间复杂度$O(n)$(?)

## [D. Present](https://codeforces.com/contest/1323/problem/D)

有一个数组，计算

$$
(a_1 + a_2) \oplus (a_1 + a_3) \oplus \ldots \oplus (a_1 + a_n) \\ \oplus (a_2 + a_3) \oplus \ldots \oplus (a_2 + a_n) \\ \ldots \\ \oplus (a_{n-1} + a_n) \\
$$

每一位来考虑，可以发现第$k$位数字只与$a_i$的第$0\sim k$位有关，也就是说和$a_i \mod 2^{k+1}$有关。取模之后$a_i$之和不会超过$2^{k+2}-2$，第$k$位为$1$当且仅当和落在$[2^k, 2^{k+1})$或$[2^{k+1}+2^k, 2^{k+2}-2]$。

## [1324D - Pair of Topics](https://codeforces.com/contest/1324/problem/D)

题目说给出$a_i, b_i$两个数组，求使得$a_i+a_j>b_i+b_j, i<j$的$(i, j)$对数。等式可以改写为$(a_i - b_i) + (a_j - b_j) > 0$，定义$c_i = a_i + b_i$，也就是求$i<j$且$c_i+c_j>0$的$(i, j)$对数。

可以先对$c$数组升序排序后枚举$j$，对每个$j$，寻找$i$使得$c_j > 0$且$c_j > -c_i$。$i$可以通过二分`std::lower_bound`找到。每找到一个，答案增加$j-i$。
