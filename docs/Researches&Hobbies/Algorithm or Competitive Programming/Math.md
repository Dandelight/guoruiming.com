|                                             |       |                                                                                          |
| ------------------------------------------- | ----- | ---------------------------------------------------------------------------------------- |
| UVa10168, Summation of Four Primes          | 提高  | https://www.luogu.com.cn/problem/UVA10168                                                |
| UVa10871, Primed Subsequence                | 普及  | https://www.luogu.com.cn/problem/UVA10871                                                |
| POJ1401 Factorial                           | 提高  | http://bailian.openjudge.cn/practice/1401/                                               |
| [NOIP2009 提高组] Hankson 的趣味题          | 提高  | https://www.luogu.com.cn/problem/P1072                                                   |
| POJ2115, C Looooops                         | 提高  | http://bailian.openjudge.cn/practice/2115/                                               |
| P4549 【模板】裴蜀定理**(要求严格证明)**    | 提高- | https://www.luogu.com.cn/problem/P4549                                                   |
| CF1514C Product 1 Modulo N                  | 提高- | https://codeforces.com/problemset/problem/1514/C                                         |
| CF1225D Power Products                      | 提高  | https://codeforces.com/problemset/problem/1225/D                                         |
| [NOIP2017 提高组] 小凯的疑惑                | 提高  | https://www.luogu.com.cn/problem/P3951                                                   |
| [NOIP2012 提高组] 同余方程                  | 提高- | https://www.luogu.com.cn/problem/P1082                                                   |
| UVA294 Divisors                             | 提高  | https://www.luogu.com.cn/problem/UVA294                                                  |
| [AHOI2005]约数研究                          | 普及- | https://www.luogu.com.cn/problem/P1403                                                   |
| Trees in a Wood, UVa10214                   | 提高  | https://www.luogu.com.cn/problem/UVA10214                                                |
| GCD Extreme（II）,UVa11426                  | 提高+ | https://www.luogu.com.cn/problem/UVA11426                                                |
| CF1536C Diluc and Kavya                     | 提高  | https://codeforces.com/problemset/problem/1536/C                                         |
| CF1349A Orac and LCM                        | 提高+ | https://codeforces.com/problemset/problem/1349/A**(要求严格证明，要求空间复杂度O(1)!!)** |
| CF1499D The Number of Pairs                 | 提高+ | https://codeforces.com/problemset/problem/1499/D                                         |
| A Horrible Poem，POI2012                    | NOI-  | https://www.luogu.com.cn/problem/P3538                                                   |
| [TJOI2009] 猜数字**(不允许快速乘法或高精)** | 提高  | https://www.luogu.com.cn/problem/P3868                                                   |
| Code Feat, UVa11754                         | 提高+ | https://www.luogu.com.cn/problem/UVA11754                                                |

## Summation of Four Primes

```cpp
// 任一大于2的偶数都可写成两个素数之和
#include <bits/stdc++.h>
using namespace std;
const int N = 10000000 + 10;
int n, cnt;
int prime[N / 2];
bool v[N];
int main() {
  v[0] = 1, v[1] = 1;
  for (int i = 1; i < N; ++i) {
    if (!v[i]) prime[++cnt] = i;
    for (int j = 1; j <= cnt && i * prime[j] < N; ++j) {
      v[i * prime[j]] = 1;
      if (i % prime[j] == 0) break;
    }
  }
  while (scanf("%d", &n) == 1) {
    if (n < 8)
      puts("Impossible.");
    else {
      if (n % 2 == 0) {
        n -= 4;
        for (int i = 1; i <= cnt; ++i) {
          if (!v[prime[i]] && !v[n - prime[i]]) {
            printf("2 2 %d %d\n", prime[i], n - prime[i]);
            break;
          }
        }
      } else {
        n -= 5;
        for (int i = 1; i <= cnt; ++i) {
          if (!v[prime[i]] && !v[n - prime[i]]) {
            printf("2 3 %d %d\n", prime[i], n - prime[i]);
            break;
          }
        }
      }
    }
  }
}
```

## Primed Subsequence

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 100000021;
ll n, cnt, prime[N / 2], t;
bool v[N];
const int M = 10010;
ll a[M], cumsum[M];
int main() {
  v[0] = v[1] = 1;
  for (int i = 2; i < N; ++i) {
    if (!v[i]) prime[++cnt] = i;
    for (int j = 1; j <= cnt && i * prime[j] < N; ++j) {
      v[i * prime[j]] = 1;
      if (i % prime[j] == 0) break;
    }
  }
  scanf("%lld", &t);
  for (int i = 1; i <= t; ++i) {
    int found = 0;
    scanf("%lld", &n);
    for (int j = 1; j <= n; ++j) scanf("%lld", a + j);
    for (int j = 1; j <= n; ++j) cumsum[j] = a[j] + cumsum[j - 1];
    for (int j = 2; j <= n; ++j) {            // 枚举区间长度
      for (int k = 1; k <= n - j + 1; ++k) {  // 枚举区间左端点
        if (v[cumsum[k + j - 1] - cumsum[k - 1]] == 0) {
          printf("Shortest primed subsequence is length %d:", j);
          for (int p = k; p <= k + j - 1; ++p) printf(" %d", a[p]);
          puts("");
          found = 1;
          break;
        }
      }
      if (found) break;
    }
    if (!found) printf("This sequence is anti-primed.\n");
  }
}
```

## Factorial

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
  int tc;
  cin >> tc;
  while (tc--) {
    ll n;
    cin >> n;
    int ans = 0;
    while (n) {
      ans += n / 5;
      n /= 5;
    }
    cout << ans << endl;
  }
}
```

## Hankson 的趣味题

```cpp
/**
 * 枚举 0 到 sqrt(b1) 中 b1 的因子(也就是 x )，
 * 如果这个数是 a1 的整数倍，并且满足
 * gcd(x/a1, a0/a1) = 1
 * gcd(b1/b0, b1/x) = 1
 * 则ans++
 *
 * ref: https://blog.csdn.net/nuclearsubmarines/article/details/77603154
 */

#include <bits/stdc++.h>
using namespace std;
int main() {
  int tc;
  cin >> tc;
  while (tc--) {
    int a0, a1, b0, b1;
    cin >> a0 >> a1 >> b0 >> b1;
    int p = a0 / a1, q = b1 / b0, ans = 0;
    for (int x = 1; x <= b1 / x; ++x) {
      if (b1 % x == 0) {
        if (x % a1 == 0 && __gcd(x / a1, p) == 1 && __gcd(q, b1 / x) == 1)
          ans++;
        int y = b1 / x;
        if (x == y) continue;
        if (y % a1 == 0 && __gcd(y / a1, p) == 1 && __gcd(q, b1 / y) == 1)
          ans++;
      }
    }
    cout << ans << endl;
  }
}
```

## 裴蜀定理

已知：$\sum_{i=1}^n a_i x_i = f$，其中$a_i, x_i, f \in Z$

求证：此方程有解的充要条件是$\gcd(a_1, a_2, \ldots, a_n) | f$，其中$n \in N^+ \and n \in [2, +\infty)$

证明：

必要性：

设$\gcd(a_1, a_2, \ldots, a_n) = k$，则$k|a_1, k|a_2, \ldots, k | a_n$

$\therefore k | \sum_{i=1}^na_ix_i$

即$\gcd(a_1, a_2, \ldots, a_n) | f$

充分性：

使用数学归纳法

显然$n=2$时成立

假设$n=x$时成立，考虑$n=x+1$时的情况

设$\sum_{i=1}^na_i = S$，$\gcd(a_1, a_2, \ldots, a_n) = k$，

总存在一对整数$(N, M)$使得$N\times S + M \times a_{x+1} = \gcd(k, a_{x+1})$

```cpp
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  int ans;
  cin >> n >> ans;
  ans = abs(ans);
  for (int i = 0; i < n - 1; ++i) {
    int x;
    cin >> x;
    x = abs(x);
    ans = __gcd(x, ans);
  }
  cout << ans << endl;
}
```

## Product 1 Modulo N

```cpp
#include <bits/stdc++.h>
using namespace std;
bool ok[100005];
// 结果中所有数字都必须要与n互质
int main() {
  int n;
  scanf("%d", &n);
  long long prod = 1;
  for (int i = 1; i < n; i++) {
    if (__gcd(n, i) == 1) {
      ok[i] = 1;
      prod = (prod * i) % n;
    }
  }
  if (prod != 1) ok[prod] = 0;
  printf("%d\n", count(ok + 1, ok + n, 1));
  for (int i = 1; i < n; i++) {
    if (ok[i]) printf("%d ", i);
  }
}
```

## Power Products

```cpp
#include <bits/stdc++.h>
using namespace std;
using LL = long long;
using VIP = vector<pair<int, int>>;
VIP factor(int n, int k) {
  map<int, int> F;
  for (int i = 2; i * i <= n; i++)
    while (n % i == 0)
      n /= i, F[i]++;
  if (n > 1)
    F[n]++;
  VIP L;
  for (const auto &p : F)
    if (p.second % k)
      L.push_back({p.first, p.second % k});
  return L;
}
int main() {
  int N, K;
  cin >> N >> K;
  map<VIP, int> FM;
  for (int i = 0, a; i < N; i++)
    cin >> a, FM[factor(a, K)]++;
  LL ans = 0;
  for (const auto &fm : FM) {
    VIP f2 = fm.first;
    LL fc = fm.second;
    for (auto &p : f2)
      p.second = K - p.second;
    if (FM.count(f2))
      ans += fm.first == f2 ? (fc * (fc - 1)) : (fc * FM[f2]);
  }
  cout << ans / 2 << endl;
}
```

## 小凯的疑惑

```cpp
#include <iostream>
using namespace std;
int main() {
	long long a, b;
	cin >> a >> b;
	cout << a * b - a - b << endl;
	return 0;
}
```

## 同余方程

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

ll gcd(ll a, ll b) { return b == 0 ? a : gcd(b, a % b); }

void ex_gcd(ll a, ll b, ll &x, ll &y) {
  if (b == 0) {
    x = 1, y = 0;
    return;
  }
  ex_gcd(b, a % b, y, x);
  y -= (a / b) * x;
}

ll a, b;
int main() {
  cin >> a >> b;
  if (a > b) swap(a, b);
  ll x, y;
  ex_gcd(a, b, x, y);
  if (x > 0) {
    swap(a, b);
    swap(x, y);
  }
  ll tmp = (-x) / b;
  x = x + tmp * b;
  y = y - tmp * a;
  while (x < 0) x = x + b, y = y - a;
  while (x > 0) x = x - b, y = y + a;
  ll ans;
  ll xx2 = x + b;
  ans = a * (xx2 - 1) + b * (y - 1);
  cout << ans - 1 << endl;

  return 0;
}
```

## Divisors

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
const int NN = 1e5 + 4;
bool isPrime[NN];
vector<int> Primes;
void sieve() {
  fill_n(isPrime, NN, true), isPrime[1] = false;
  for (LL i = 2; i < NN; i++) {
    if (!isPrime[i])
      continue;
    Primes.push_back(i);
    for (LL j = i * i; j < NN; j += i)
      isPrime[j] = false;
  }
}
LL divCnt(int x) {
  LL c = 1, xb = x;
  for (size_t i = 0; i < Primes.size() && Primes[i] < xb; i++) {
    int p = Primes[i], pc = 0;
    while (x % p == 0)
      x /= p, ++pc;
    c *= pc + 1;
  }
  if (x > 1)
    c *= 2;
  return c;
}
int main() {
  ios::sync_with_stdio(false), cin.tie(0);
  sieve();
  int T;
  cin >> T;
  for (int L, U; T--;) {
    cin >> L >> U;
    LL ans = 1, ans_c = -1;
    for (int x = L; x <= U; x++) {
      LL c = divCnt(x);
      if (c > ans_c)
        ans = x, ans_c = c;
    }
    printf("Between %d and %d, %lld has a maximum of %lld divisors.\n", L, U,
           ans, ans_c);
  }
  return 0;
}
```

## 约数研究

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
int main() {
  ios::sync_with_stdio(false), cin.tie(0);
  int n;
  cin >> n;
  LL ans = 0;
  for (int i = 1; i <= n; i++)
    ans += n / i; // i的倍数有n/i个,是 n/i个数字的约数
  cout << ans << endl;
  return 0;
}
```

## 森林里的树

```cpp
#include <bits/stdc++.h>
using namespace std;
#define _for(i, a, b) for (int i = (a); i < (b); ++i)
typedef long long LL;
int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }
const int MAXA = 2000 + 4;
LL Phi[MAXA];
void init() {
  fill_n(Phi, MAXA, 0);
  Phi[1] = 1;
  _for(i, 2, MAXA) if (Phi[i] == 0) {
    for (LL j = i; j < MAXA; j += i) {
      LL &pj = Phi[j];
      if (pj == 0)
        pj = j;
      pj = pj / i * (i - 1);
    }
  }
}
int main() {
  init();
  for (int A, B; scanf("%d%d", &A, &B) == 2 && A && B;) {
    LL P = 0;
    for (int x = 1; x <= A; x++) {
      int k = B / x;
      P += k * Phi[x];
      for (int y = k * x + 1; y <= B; y++)
        if (gcd(x, y) == 1)
          P++;
    }
    double ans = 4 * (P + 1);
    LL N = 4LL * A * B + 2LL * A + 2LL * B;
    printf("%.7lf\n", ans / N);
  }
  return 0;
}
```

## GCD Extreme

```cpp
#include <cstdio>
#include <cstring>
const int NN = 4000000;
typedef long long LL;
int phi[NN + 1];
void phi_table(int n) {
  for (int i = 2; i <= n; i++)
    phi[i] = 0;
  phi[1] = 1;
  for (int i = 2; i <= n; i++)
    if (!phi[i])
      for (int j = i; j <= n; j += i) {
        if (!phi[j])
          phi[j] = j;
        phi[j] = phi[j] / i * (i - 1);
      }
}
LL S[NN + 1], f[NN + 1];
int main() {
  phi_table(NN);
  memset(f, 0, sizeof(f)); // !"#f
  for (int i = 1; i <= NN; i++)
    for (int n = i * 2; n <= NN; n += i)
      f[n] += i * phi[n / i];
  S[2] = f[2]; // !"#S
  for (int n = 3; n <= NN; n++)
    S[n] = S[n - 1] + f[n];
  for (int n; scanf("%d", &n) == 1 && n;)
    printf("%lld\n", S[n]);
  return 0;
}
```

## Diluc and Kavya

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
using IPair = pair<int, int>;
LL gcd(LL a, LL b) { return b == 0 ? a : gcd(b, a % b); }
int main() {
  ios::sync_with_stdio(false), cin.tie(0);
  int T, N;
  cin >> T;
  string S;
  while (T--) {
    cin >> N >> S;
    map<IPair, int> M;
    for (int i = 0, k = 0, d = 0; i < N; i++) {
      k += S[i] == 'K', d += S[i] == 'D';
      int g = gcd(k, d);
      cout << ++M[{k / g, d / g}] << " ";
    }
    cout << endl;
  }
  return 0;
}
```

## Orac and LCM

```cpp
#include <bits/stdc++.h>
using namespace std;
using LL = long long;
LL gcd(LL a, LL b) { return b == 0 ? a : gcd(b, a % b); }
LL ga, s;
void update() {
  LL g = gcd(ga, s);
  s = ga / g * s, ga = g;
}
int main() {
  ios::sync_with_stdio(false), cin.tie(0);
  int n;
  cin >> n >> ga >> s, update();
  for (int i = 2, a; i < n; i++)
    cin >> a, s = gcd(s, a), update();
  cout << s << endl;
}
```

## The number of pairs

```cpp
#include <bits/stdc++.h>
using namespace std;
const int K = 2e7 + 4;
int P[K], C[K];
int main() {
  for (int i = 2; i < K; ++i) { // 筛法,线性筛加速
    int &p = P[i];              // 最小素因子
    if (p == 0)                 // 素数
      for (int j = i; j < K; j += i)
        if (P[j] == 0)
          P[j] = i;
  }
  for (int k = 2; k < K; ++k) {
    int p = P[k], j = k / p;
    C[k] = C[j] + (p != P[j]);
  }
  int t, c, d, x, ans;
  scanf("%d", &t);
  while (t--) {
    scanf("%d%d%d", &c, &d, &x), ans = 0;
    for (int g = 1; g * g <= x; ++g)
      if (x % g == 0) { // 枚举x的约数g
        int k = x / g + d;
        if (k % c == 0)
          ans += 1 << C[k / c];
        if (g * g == x)
          continue;
        k = g + d;
        if (k % c == 0)
          ans += 1 << C[k / c];
      }
    printf("%d\n", ans);
  }
}
```

## A Horrible Poem

```cpp
#include <bits/stdc++.h>
using namespace std;
const int NN = 5e5 + 4, x = 263;
typedef unsigned long long ULL;
typedef long long LL;
ULL XP[NN];
void initXP() {
  XP[0] = 1;
  for (size_t i = 1; i < NN; i++)
    XP[i] = x * XP[i - 1];
}
template <size_t SZ> struct StrHash {
  size_t N;
  ULL H[SZ];
  void init(const char *pc, size_t n = 0) {
    if (XP[0] != 1)
      initXP();
    if (n == 0)
      n = strlen(pc);
    N = n, H[N] = 0;
    for (int i = N - 1; i >= 0; --i)
      H[i] = pc[i] - 'a' + 1 + x * (H[i + 1]);
  }
  void init(const string &S) { init(S.c_str(), S.size()); }
  inline ULL hash(size_t i, size_t j) { // hash[i, j]
    return H[i] - H[j + 1] * XP[j - i + 1];
  }
  inline ULL hash() { return H[0]; }
};
StrHash<NN> hs;
char S[NN];
int lastP[NN], primes[NN], pCnt;
void sieve(int N) {
  pCnt = 0;
  fill_n(lastP, N, 0);
  int *P = primes;
  for (int i = 2; i < N; ++i) {
    int &l = lastP[i]; // i的最小素因子
    if (l == 0)
      l = i, P[pCnt++] = i; // i是素数
    for (int j = 0; j < pCnt && P[j] <= l && P[j] * i < N; ++j)
      lastP[i * P[j]] = P[j]; // i*p的最小素因子是p
  }
}
int find_rep(int a, int b) {
  int L = b - a + 1, xl = L;
  while (xl > 1) {
    int p = lastP[xl]; // 尝试每一个素因子
    if (hs.hash(a, b - L / p) == hs.hash(a + L / p, b))
      L /= p;
    xl /= p;
  }
  return L;
}
int main() {
  int n, q;
  S[0] = '|';
  scanf("%d%s%d", &n, S + 1, &q);
  hs.init(S, n + 1), sieve(n + 1);
  for (int i = 0, a, b; i < q; i++)
    scanf("%d%d", &a, &b), printf("%d\n", find_rep(a, b));
  return 0;
}
```

## 猜数字

```cpp
#include <bits/stdc++.h>
using namespace std;
using LL = long long;
#define _all(i, a, b) for (int i = (a); i <= (int)(b); ++i)
int N;
LL A[20], B[20];
void exgcd(LL a, LL b, LL &x, LL &y) {
  b == 0 ? (x = 1, y = 0) : (exgcd(b, a % b, y, x), y -= a / b * x);
}
LL mul(initializer_list<LL> xs, LL mod) {
  unsigned long long m = 1;
  for (LL x : xs)
    (m *= x) %= mod;
  return m;
}
int main() {
  ios::sync_with_stdio(false), cin.tie(0);
  cin >> N;
  _all(i, 1, N) cin >> A[i];
  LL ans = 0, M = 1;
  _all(i, 1, N) {
    LL &b = B[i], &a = A[i];
    cin >> b, a = (a % b + b) % b, M *= b;
  }
  for (int i = 1; i <= N; ++i) { // 中国剩余定理
    LL b = B[i], w = M / B[i], x, y;
    exgcd(w, b, x, y), x = (x % b + b) % b;
    (ans += mul({w, x, A[i]}, M)) %= M;
  }
  cout << ans << endl;
  return 0;
}
```
