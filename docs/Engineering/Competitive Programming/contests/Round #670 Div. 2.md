## [1406C - Link Cut Centroids](https://codeforces.com/contest/1406/problem/C)

结论：树有一个或两个重心，如果有两个重心，那么它们一定直接相连。

那就好说了，如果只有一个重心，随便断一条边连上；如果有两个，设为$cent_1$和$cent_2$，那么找到$cent_2$所在子树上摘一个叶子连到$cent_1$上。

```cpp
#include <bits/stdc++.h>
using namespace std;
#define rep(i, a, b) for (int i = a; i < b; ++i)
const int N = 100010;
int n, sz[N], fa[N], minn = 1e9, cent1, cent2;
vector<int> g[N];
int S;
void dfs(int x, int f) {
  fa[x] = f, sz[x] = 1;
  int mx = 0;
  for (int y : g[x]) {
    if (y == f) continue;
    dfs(y, x);
    sz[x] += sz[y];
    mx = max(mx, sz[y]);
  }
  mx = max(mx, n - sz[x]);
  if (mx < minn)
    minn = mx, cent1 = x, cent2 = 0;
  else if (mx == minn)
    cent2 = x;
}
void dfs2(int x, int f) {
  if (g[x].size() == 1) {
    S = x;
    return;
  }
  for (int y : g[x]) {
    if (y == f) continue;
    dfs2(y, x);
  }
}
int main() {
  int tc;
  cin >> tc;
  while (tc--) {
    cin >> n;
    cent1 = cent2 = 0, minn = 1e9;
    rep(i, 1, n + 1) g[i].clear(), fa[i] = 0;
    rep(i, 1, n) {
      int u, v;
      cin >> u >> v;
      g[u].push_back(v);
      g[v].push_back(u);
    }
    dfs(1, 0);
    if (!cent2) {
      printf("1 %d\n1 %d\n", g[1][0], g[1][0]);
      continue;
    }
    if (fa[cent1] != cent2) swap(cent1, cent2);
    dfs2(cent1, cent2);
    printf("%d %d\n%d %d\n", S, fa[S], S, cent2);
  }
}
```

## [1406D - Three Sequences](https://codeforces.com/contest/1406/problem/D)

给定数组$a$，需要寻找两个数组$b, c$，使得

1. $b_1 + c_i = a_i$
2. $b$单调不下降
3. $c$单调不上升

需要最小化$\max(b_i, c_i)$。

之后还会有$q$个改变，每次给出三个整数$l, r, x$，将$a$数组中$[l, r]$之间的数字增加$x$。

经过简单的推导之后发现只需要最小化$\max(b_n, c_1)$即可。

可以发现

$$
\begin{cases}
\left \{
\begin{aligned}
b_i &= b_{i-1}+a_i-a_{i-1} \\
c_i &= c_{i-1}
\end{aligned}
\right. & \text{if}\quad a_i > a_{i-1} \\
\left \{
\begin{aligned}
b_i &= b_{i-1} \\
c_i &= c_{i-1}+a_i-a_{i-1}
\end{aligned}
\right. & \text{if}\quad a_i < a_{i-1} \\
\end{cases}
$$

设$K = \sum\limits_{i=2}^{n}\max(0,a_i-a_{i-1})$，设$c_1 = x$，则$b_1 = a_1 - x$，$b_n = b_1 + K = a_1 - x + K$。

需要最小化$\max(x, a_1-x+K)$，初始$x$应设置为$\dfrac{a_1 + K}{2}$。注意当$a_1+K$是奇数时，$|c_1 - b_n| = 1$，也就是一个会比另一个大出$1$。

接下来的$q$次询问只会改变$\sum\max(0,a_i-a_{i-1})$，所以只有$a_l - a_{l-1}$和$a_{r+1}-a_r$需要改变。注意当$l=1$时$a_1$也要进行改变。

每次输出$\max(x, a_1-x+K)$即可。
