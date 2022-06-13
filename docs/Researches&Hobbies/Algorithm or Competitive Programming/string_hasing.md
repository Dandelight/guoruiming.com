| 2h   | [CEOI2017]Palindromic Partitions | https://www.luogu.com.cn/problem/P4656     | 提高+ |
| ---- | -------------------------------- | ------------------------------------------ | ----- |
| 2h   | [USACO16 JAN G]Lights Out        | https://www.luogu.com.cn/problem/P3134     | 提高+ |
| 2h   | [COCI2016-2017#4] Osmosmjerka    | https://www.luogu.com.cn/problem/P7538     | NOI-  |
| 选做 | [COCI2020-2021#3] Sateliti       | https://www.luogu.com.cn/problem/P7170     | NOI   |
| 选做 | Baltic OI 2018 – Genetics        | https://open.kattis.com/problems/genetics2 | NOI-  |

## Palindromic Partitions

从两边往中间推，假设两边原来是 $\mathrm{a} \alpha A\beta\mathrm{a}$，那么同时把左右的 $\mathrm a$  分出来，答案就增加了 $2$。利用字符串哈希可以方便地做到。

```cpp
#include <cstdio>
#include <cstring>
typedef unsigned long long ull;
const int P = 79;
int T, N;
char str[1000005];

int main() {
  scanf("%d", &T);
  while (T--) {
    scanf("%s", str);
    N = strlen(str);
    ull s1 = 0, s2 = 0, p = 1;
    int ans = 0;
    for (int i = 0; i < N / 2; ++i) {
      s1 = s1 * P + str[i];
      s2 = str[N - 1 - i] * p + s2;
      p = p * P;
      if (s1 == s2) {
        ans += 2;
        s1 = s2 = 0;
        p = 1;
      }
    }
    if (N % 2 || s1) ++ans;
    printf("%d\n", ans);
  }
  return 0;
}
```

## Lights out

```cpp
/**
 * @file luogu/3131/main
 * @brief
 * @see
 * @author Ruiming Guo (guoruiming@stu.scu.edu.cn)
 * @copyright 2022
 * @date 2022/6/5 00:28:06
 **/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <queue>
typedef long long ll;
using namespace std;

const ll mod = 2005060711;
int n, hs[205][205],  // hs[]为hash数组，注意此处未将初始位置i的转角度数进行存储
    expTab[805], d1[205], d2[205], b[205], nxt[205], tot, ans;
struct point {
  int x, y;
} t[205];
int read() {
  int x = 0, f = 1;
  char ch = getchar();
  while (ch > '9' || ch < '0') {
    if (ch == '-') f = -1;
    ch = getchar();
  }
  while (ch >= '0' && ch <= '9') {
    x = (x << 3) + (x << 1) + ch - '0';
    ch = getchar();
  }
  return x * f;
}
// 当前 Bessie 站在 x 处，背对 x-1 ，准备向 x+1 前进需要向左(0)或是向右(1)转
ll cross(point p1, point p2, point p0) {
  return 1LL * (p1.x - p0.x) * (p2.y - p0.y) <
         1LL * (p2.x - p0.x) * (p1.y - p0.y);
}
int dis(int a, int b) {
  if (t[a].x == t[b].x) return abs(t[a].y - t[b].y);
  return abs(t[a].x - t[b].x);
}
int ask(int x) {
  int s = lower_bound(b + 1, b + 1 + tot, x) - b;
  int a = 0;
  while (s) s /= 10;
  return a;
}
int main() {
  n = read();
  expTab[0] = 1;
  for (int i = 1; i <= 800; ++i) expTab[i] = 13LL * expTab[i - 1] % mod;
  for (int i = 1; i <= n; ++i) {
    int x = read(), y = read();
    t[i] = point{x, y};
  }
  t[n + 1] = t[1];
  for (int i = 2; i <= n; ++i) b[++tot] = dis(i, i - 1);
  sort(b + 1, b + tot);
  tot = unique(b + 1, b + n) - b - 1;

  for (int i = 2; i <= n; ++i) d2[i] = d2[i - 1] + dis(i, i - 1);
  for (int i = n; i >= 2; --i) d1[i] = d1[i + 1] + dis(i, i + 1);

  for (int i = 2; i <= n; ++i) nxt[i] = cross(t[i], t[i + 1], t[i - 1]);
  for (int i = 2; i <= n; ++i) {
    for (int j = i + 1; j <= n + 1; ++j) {
      ll x = dis(j - 1, j) * 13LL % mod + nxt[j];
      x = (x >= mod) ? x - mod : x;
      hs[i][j] =
          (hs[i][j - 1] * expTab[ask(dis(j - 1, j)) + 1] % mod + x) % mod;
    }
  }
  for (int i = 2; i <= n; ++i) {
    if (d1[i] <= d2[i])
      continue;  // 若i的较短路方向即为顺时针，则一定不会产生多余路程
    for (int j = i + 1; j <= n + 1; ++j) {
      int s = 0;
      for (int k = 2; k <= n; ++k) {
        if (nxt[k] == nxt[i] && hs[k][j - i + k] == hs[i][j]) s++;
      }
      if (s == 1) {
        ans = max(ans, d1[i] - d1[j] + min(d2[j], d1[j]) - d2[i]);
        break;
      }
    }
  }
  cout << ans << endl;
  return 0;
}
```
