# **20220327-线性代数**

|       |            |                                    |            |       |                                                 |
| ----- | ---------- | ---------------------------------- | ---------- | ----- | ----------------------------------------------- |
| 0.5h  | 矩阵乘法   | [TJOI2017]可乐                     | 图上路径数 | 提高  | https://www.luogu.com.cn/problem/P3758          |
| 1h    | 矩阵乘法   | Recurrences,UVa10870               | 伴随矩阵   | 提高+ | https://www.luogu.com.cn/problem/UVA10870       |
| 1h    | 矩阵乘法   | USACO2007 Nov G. Cow Relays        | Floyd 加速 | 提高+ | https://www.luogu.com.cn/problem/P2886          |
| 1h    | 矩阵乘法   | [2020-NOI Online #3 提高组] 魔法值 | DP 加速    | 提高+ | https://www.luogu.com.cn/problem/P6569          |
| 0.5h  | 矩阵乘法   | CF691E Xor-sequences               | DP 加速    | 提高  | https://codeforces.com/contest/691/problem/E    |
| 0.5h  | 高斯消元   | [JSOI2008]球形空间产生器           | 方程转换   | 提高+ | https://www.luogu.com.cn/problem/P4035          |
| 0.5h  | 矩阵乘法   | [TJOI2019]甲苯先生的字符串         | 路径统计   | 提高+ | https://www.luogu.com.cn/problem/P5337          |
| 0.75h | 高斯消元   | 乘积是平方数(Square, UVa11542)     | 异或方程   | NOI-  | https://www.luogu.com.cn/problem/UVA11542       |
| 2h    | 矩阵线段树 | CF718C. Sasha and Array            | 斐波那契   | NOI-  | https://codeforces.com/problemset/problem/718/C |
| 1h    | 矩阵乘法   | Balkan OI 2009 - Reading           | 分层图     | NOI-  | https://www.luogu.com.cn/problem/P6841          |
| 0.25h | 高斯消元   | [JSOI2008]球形空间产生器           | 方程转换   | 提高+ | https://www.luogu.com.cn/problem/P4035          |
| 1h    | 高斯消元   | [SDOI2010] 外星千足虫              | 异或方程   | 提高+ | https://www.luogu.com.cn/problem/P2447          |
| 1h    | 矩阵乘法   | Recurrences,UVa10870               | 伴随矩阵   | 提高+ | https://www.luogu.com.cn/problem/UVA10870       |

## 可乐

```cpp
#include <bits/stdc++.h>
using namespace std;
const int MOD = 2017;
typedef long long ll;
typedef vector<int> vi;
struct Mat {
  int r, c;
  vector<vi> f;
  Mat(int _r, int _c) : r(_r), c(_c), f(r + 1, vi(c + 1)) {}
  Mat operator*(const Mat &b) const {
    Mat m(r, b.c);
    for (int i = 1; i <= m.r; ++i)
      for (int j = 1; j <= m.c; ++j)
        for (int k = 1; k <= m.c; ++k)
          (m.f[i][j] += f[i][k] * b.f[k][j] % MOD) %= MOD;
    return m;
  }
  Mat operator^(ll x) const {
    assert(r == c);
    Mat m = *this, p = *this;
    for (--x; x; x >>= 1, p = p * p)
      if (x & 1) m = m * p;
    return m;
  }
};
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  int n, m, t;
  cin >> n >> m;
  ++n;
  Mat G(n, n), A(1, n);
  for (int u, v, i = 1; i <= m; ++i) cin >> u >> v, G.f[u][v] = G.f[v][u] = 1;

  for (int i = 1; i <= n; ++i) G.f[i][i] = 1;
  for (int i = 1; i < n; ++i) G.f[i][n] = 1;
  cin >> t;
  A.f[1][1] = 1, A = A * (G ^ t);
  int ans = 0;
  for (int i = 1; i <= n; ++i) (ans += A.f[1][i]) %= MOD;
  cout << ans << endl;
  return 0;
}
```

## Recurrences(未完成)

## Cow Relays

很容易想到 N 次矩阵乘法的思路，不同的是这次的是长度而不是方案数（所以是加法而不是乘法）、是最小值而不是总数，所以把`*`换成`+`，把`+`换成`min`。

```cpp
#include <bits/stdc++.h>
using namespace std;
#define rep(i, a, b) for (int i = (a); i < (int)b; ++i)
const int MV = 205, INF = 0x7f7f7f7f;
typedef int Mat[MV][MV];
int sz;
void matmul(Mat A, Mat B, Mat C) {
  static Mat m;
  memset(m, 0x7f, sizeof(Mat));
  rep(i, 0, sz) rep(j, 0, sz) rep(k, 0, sz) {
    if (A[i][k] != INF && B[k][j] != INF)
      m[i][j] = min(m[i][j], A[i][k] + B[k][j]);
  }
  memcpy(C, m, sizeof(Mat));
}
void matpow(Mat A, int n, Mat r) {
  Mat a;
  memcpy(a, A, sizeof(a));
  memcpy(r, A, sizeof(Mat));
  --n;
  for (int i = 0; (1 << i) <= n; ++i) {
    if (n & (1 << i)) matmul(r, a, r);
    matmul(a, a, a);
  }
}
map<int, int> ID;
int getId(int u) {
  if (!ID.count(u)) ID[u] = sz++;
  return ID[u];
}
Mat G, Ans;
int main() {
  int N, T, S, E;
  cin >> N >> T >> S >> E;
  sz = 0, memset(G, 0x7f, sizeof(G));
  for (int i = 1, u, v, w; i <= T; ++i) {
    cin >> w >> u >> v;
    u = getId(u), v = getId(v);
    G[u][v] = G[v][u] = min(G[u][v], w);
  }
  matpow(G, N, Ans);
  cout << Ans[getId(S)][getId(E)] << endl;
}
```

## 魔法值

```cpp
#include <bits/stdc++.h>
#define rep(i, a, b) for (int i = (a); i < (b); ++i)
using namespace std;
typedef long long ll;
typedef vector<ll> vl;
struct Mat {
  int r, c;
  vector<vl> f;
  Mat(int r, int c) : r(r), c(c), f(r, vl(c)) {}
  Mat operator*(const Mat &b) {
    Mat m(r, b.c);
    rep(i, 0, m.r) rep(j, 0, m.c) rep(k, 0, c) m.f[i][j] ^= f[i][k] * b.f[k][j];
    return m;
  }
};
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  const int BT = 32;
  int n, m, q;
  cin >> n >> m >> q;
  Mat f0(1, n);
  vector<Mat> E(BT, Mat(n, n));
  rep(i, 0, n) cin >> f0.f[0][i];
  for (int i = 0, u, v; i < m; ++i) {
    cin >> u >> v;
    --u, --v;
    E[0].f[u][v] = E[0].f[v][u] = 1;
  }
  rep(i, 1, BT) E[i] = E[i - 1] * E[i - 1];
  Mat f(1, n);
  for (ll a = 0; q; --q) {
    Mat f = f0;
    cin >> a;
    for (int i = 0; (1LL << i) <= a; ++i) {
      if (a & (1LL << i)) f = f * E[i];
    }
    cout << f.f[0][0] << endl;
  }
}
```

## 球形空间产生器

```cpp
#include <bits/stdc++.h>
using namespace std;
const double eps = 1e-8;
// b: 系数矩阵，c: 常数
double a[20][20], b[20], c[20][20];
int n;
int main() {
  cin >> n;
  for (int i = 1; i <= n + 1; ++i)
    for (int j = 1; j <= n; ++j) scanf("%lf", &a[i][j]);
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= n; ++j) {
      c[i][j] = 2 * (a[i][j] - a[i + 1][j]);
      b[i] += a[i][j] * a[i][j] - a[i + 1][j] * a[i + 1][j];
    }
  // 高斯消元
  for (int i = 1; i <= n; ++i) {
    // 找到 x[i] 的系数不为 0 的一个方程
    for (int j = i; j <= n; ++j) {
      if (fabs(c[j][i]) > eps) {
        for (int k = 1; k <= n; ++k) swap(c[i][k], c[j][k]);
        swap(b[i], b[j]);
      }
    }
    // 消去其它方程 x[i] 的系数
    for (int j = 1; j <= n; ++j) {
      if (i == j) continue;
      double rate = c[j][i] / c[i][i];
      for (int k = i; k <= n; ++k) c[j][k] -= c[i][k] * rate;
      b[j] -= b[i] * rate;
    }
  }
  for (int i = 1; i <= n; ++i) printf("%.3f ", b[i] / c[i][i]);
  printf("\n");
}
```

## 甲苯先生的字符串

```cpp
#include <bits/stdc++.h>
using namespace std;
const int MOD = 1e9 + 7, SG = 26;
typedef long long ll;
#define rep(i, a, b) for (int i = (a); i < (b); ++i)

struct Mat {
  ll f[SG][SG];
  Mat() { memset(f, 0, sizeof(f)); }
  Mat operator*(const Mat &b) const {
    Mat m;
    rep(i, 0, SG) rep(j, 0, SG) rep(k, 0, SG) {
      (m.f[i][j] += f[i][k] * b.f[k][j] % MOD) %= MOD;
    }
    return m;
  }
};
Mat matpow(Mat a, ll b) {
  Mat m;
  rep(i, 0, SG) m.f[i][i] = 1;
  for (; b; a = a * a, b >>= 1) {
    if (b & 1) m = m * a;
  }
  return m;
}
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  Mat G, A, SumG;
  rep(i, 0, SG) {
    A.f[0][i] = SumG.f[i][0] = 1;
    rep(j, 0, SG) G.f[i][j] = 1;
  }
  ll n;
  string s;
  cin >> n >> s;
  rep(i, 1, s.size()) G.f[s[i - 1] - 'a'][s[i] - 'a'] = 0;
  cout << (A * matpow(G, n - 1) * SumG).f[0][0] << endl;
}
```

## Square

```cpp
#include <bits/stdc++.h>
using namespace std;
const int maxn = 110, maxp = 510;
int vis[maxn], prime[maxp];
int get_primes(int n) {
  int m = (int)sqrt(n + 0.5);
  memset(vis, 0, sizeof(vis));
  for (int i = 2; i <= m; ++i)
    if (!vis[i])
      for (int j = i * i; j <= n; j += i) vis[j] = 1;
  int c = 0;
  for (int i = 2; i <= n; ++i)
    if (!vis[i]) prime[c++] = i;
  return c;
}
typedef int Mat[maxn][maxn];
int rank_(Mat A, int m, int n) {
  int i = 0, j = 0, k, r, u;
  while (i < m && j < n) {
    r = i;
    for (k = i; k < m; ++k)
      if (A[k][j]) {
        r = k;
        break;
      }
    if (A[r][j]) {
      if (r != i)
        for (k = 0; k <= n; ++k) swap(A[r][k], A[i][k]);
      for (int u = i + 1; u < m; ++u)
        if (A[u][j])
          for (k = i; k <= n; ++k) A[u][k] ^= A[i][k];
      i++;
    }
    j++;
  }
  return i;
}
Mat A;
int main() {
  int m = get_primes(500);
  int T;
  cin >> T;
  while (T--) {
    int n, maxp = 0;
    long long x;
    cin >> n;
    memset(A, 0, sizeof(A));
    for (int i = 0; i < n; ++i) {
      cin >> x;
      for (int j = 0; j < m; ++j)
        while (x % prime[j] == 0) {
          maxp = max(maxp, j);
          x /= prime[j];
          A[j][i] ^= 1;
        }
    }
    int r = rank_(A, maxp + 1, n);
    cout << (1LL << (n - r)) - 1 << endl;
  }
}
```

## Sasha and Array(未完成)

## Reading

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
#define rep(i, a, b) for (int i = (a); i < (int)(b); ++i)
const int MOD = 1e9 + 7, SIG = 26, FF = 5, SZ = SIG * FF + 1;
typedef vector<ll> vl;
struct Mat {
  int r, c;
  vector<vl> f;
  Mat(int r, int c) : r(r), c(c), f(r, vl(c)){};
  Mat operator*(const Mat& b) const {
    Mat m(r, b.c);
    rep(i, 0, m.r) rep(j, 0, m.c) rep(k, 0, m.c) {
      (m.f[i][j] += f[i][k] * b.f[k][j] % MOD) %= MOD;
    }
    return m;
  }
  Mat pow(ll b) const {
    Mat m = *this, p = *this;
    for (--b; b; p = p * p, b >>= 1) {
      if (b & 1) m = m * p;
    }
    return m;
  }
};
int main() {
  Mat G(SZ, SZ);
  rep(i, 0, SIG) {
    rep(j, 0, SIG) G.f[i * FF][j * FF] = 1;
    rep(f, 0, FF - 1) G.f[i * FF + f + 1][i * FF + f] = 1;
  }
  int s = SIG;
  rep(i, 0, SIG) G.f[s * FF][i * FF] = 1;
  G.f[s * FF][s * FF] = 1;
  int n, m, ans = 0;
  cin >> n >> m;
  for (int i = 0, f; i < m; ++i) {
    char a, b;
    cin >> a >> b >> f;
    int i1 = (a - 'a') * FF, i2 = (b - 'a') * FF;
    G.f[i1][i2] = 0, G.f[i1][i2 + f - 1] = 1;
    G.f[i2][i1] = 0, G.f[i2][i1 + f - 1] = 1;
  }
  G = G.pow(n + 1);
  rep(i, 0, SIG)(ans += G.f[SIG * FF][i * FF]) %= MOD;
  cout << ans << endl;
}
```

## 球形空间产生器

```cpp
#include <bits/stdc++.h>
using namespace std;
const double eps = 1e-8;
// b: 系数矩阵，c: 常数
double a[20][20], b[20], c[20][20];
int n;
int main() {
  cin >> n;
  for (int i = 1; i <= n + 1; ++i)
    for (int j = 1; j <= n; ++j) scanf("%lf", &a[i][j]);
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= n; ++j) {
      c[i][j] = 2 * (a[i][j] - a[i + 1][j]);
      b[i] += a[i][j] * a[i][j] - a[i + 1][j] * a[i + 1][j];
    }
  // 高斯消元
  for (int i = 1; i <= n; ++i) {
    // 找到 x[i] 的系数不为 0 的一个方程
    for (int j = i; j <= n; ++j) {
      if (fabs(c[j][i]) > eps) {
        for (int k = 1; k <= n; ++k) swap(c[i][k], c[j][k]);
        swap(b[i], b[j]);
      }
    }
    // 消去其它方程 x[i] 的系数
    for (int j = 1; j <= n; ++j) {
      if (i == j) continue;
      double rate = c[j][i] / c[i][i];
      for (int k = i; k <= n; ++k) c[j][k] -= c[i][k] * rate;
      b[j] -= b[i] * rate;
    }
  }
  for (int i = 1; i <= n; ++i) printf("%.3f ", b[i] / c[i][i]);
  printf("\n");
}
```

## 外星千足虫

```cpp
#include <bits/stdc++.h>
#define rep(i, a, b) for (int i = (a); i <= (int)(b); ++i)

using namespace std;
const int N = 1004;
bitset<N> Mat[2 * N];
int gauss(int n, int m) {
  int k = -1;
  rep(i, 1, n) {
    int cur = i;
    while (cur <= m && !Mat[cur][i]) cur++;
    if (cur > m) return 0;
    k = max(k, cur);
    if (cur != i) swap(Mat[cur], Mat[i]);
    rep(j, 1, m) if (i != j && Mat[j].test(i)) Mat[j] ^= Mat[i];
  }
  return k;
}
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  int n, m;
  cin >> n >> m;
  string s;
  for (int i = 1, b; i <= m; ++i) {
    cin >> s >> b;
    rep(j, 1, n) Mat[i].set(j, s[j - 1] == '1');
    Mat[i].set(0, b);
  }
  int k = gauss(n, m);
  if (k) {
    cout << k << endl;
    rep(i, 1, n) cout << (Mat[i][0] ? "?y7M#" : "Earth") << endl;

  } else {
    cout << "Cannot Determine\n";
  }
}
```
