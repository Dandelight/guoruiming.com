| 0.2h  | 组合数       | NOIP2011 提高组 计算系数        | 普及  | https://www.luogu.com.cn/problem/P1313               |
| ----- | ------------ | ------------------------------- | ----- | ---------------------------------------------------- |
| 0.25h | 组合数       | NOIP2016 提高组 组合数问题      | 提高  | https://www.luogu.com.cn/problem/P2822               |
| 0.25h | 乘法原理     | [HNOI2008]越狱                  | 提高  | https://www.luogu.com.cn/problem/P3197               |
| 0.5h  | 多重排列     | POJ3421 X-factor Chains         | 提高  | http://poj.org/problem?id=3421                       |
| 1h    | 错位排列     | [SDOI2016]排列计数              | 提高  | https://www.luogu.com.cn/problem/P4071               |
| 0.5h  | 重复选择问题 | UVa10910 Marks Distribution     | 提高  | https://www.luogu.com.cn/problem/UVA10910            |
| 0.5h  | 递推计数     | UVA1510 Neon Sign               | 提高  | https://www.luogu.com.cn/problem/UVA1510             |
| 1.5h  | 容斥原理     | UVa11806 Cheerleaders           | 提高+ | https://www.luogu.com.cn/problem/UVA11806            |
| 0.5h  | 递推计数     | UVa11401 Triangle Counting      | 提高  | https://www.luogu.com.cn/problem/UVA11401            |
| 1h    | 二项式系数   | Irrelevant Elements, NEERC 2004 | 提高  | http://bailian.openjudge.cn/practice/2167?lang=en_US |

## 计算系数

```cpp
#include <cstdio>
using namespace std;
const int N = 1005;
const int mod = 10007;
#define int long long
int c[N][N];
int a, b, k, n, m;
int pow(int x, int y) {
  int ans = 1, pas = x;
  while (y) {
    if (y & 1) ans = ans * pas % mod;
    pas = pas * pas % mod;
    y >>= 1;
  }
  return ans;
}
int dfs(int n, int m) {
  if (!m) return c[n][m] = 1;
  if (m == 1) return c[n][m] = n;
  if (c[n][m]) return c[n][m];
  if (n - m < m) m = n - m;
  return c[n][m] = (dfs(n - 1, m) + dfs(n - 1, m - 1)) % mod;
}
signed main() {
  scanf("%lld%lld%lld%lld%lld", &a, &b, &k, &n, &m);
  c[1][0] = c[1][1] = 1;
  a %= mod, b %= mod;
  int ans = 1;
  ans = (ans * pow(a, n)) % mod;
  ans = (ans * pow(b, m)) % mod;
  if (n > m) n = m;
  ans = (ans * dfs(k, n)) % mod;
  printf("%lld\n", ans);
}
```

## 组合数问题

```cpp
#include <bits/stdc++.h>
using namespace std;
int t, k, n, m;
const int N = 2005;
int c[N][N], s[N][N];
void prepare() {
  // 归纳基础
  c[1][1] = 1;
  for (int i = 0; i <= 2000; ++i) c[i][0] = 1;
  // 递推组合数
  for (int i = 2; i <= 2000; ++i) {
    for (int j = 1; j <= i; ++j) {
      c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % k;
    }
  }
  for (int i = 2; i <= 2000; ++i) {
    for (int j = 1; j <= i; ++j) {
      // 容斥原理，矩阵求和
      s[i][j] = s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1];
      // 如果是整除，答案增加1
      if (c[i][j] == 0) s[i][j] += 1;
    }
    // 向后更新一步
    s[i][i + 1] = s[i][i];
  }
}
int main() {
  cin >> t >> k;
  prepare();
  while (t--) {
    cin >> n >> m;
    if (m > n) m = n;
    cout << s[n][m] << endl;
  }
  return 0;
}
```

## 越狱

```cpp
// 考虑不会发生“越狱”的情况
// 第一个位置有 m 个选择
// 第二个位置有 (m-1) 个选择
// 以后每个位置都有 (m-1) 个选择
// 所以不会“越狱”有 m*(m-1)^(n-1) 种情况
// 一共会有 m^n 种情况
// 答案要求会“越狱”的情况，即 m^n - m*(m-1)^(n-1)

#include <iostream>
using namespace std;
typedef long long ll;
const int MOD = 100003;
ll qexp(ll base, ll power) {
  if (power == 0) return 1;
  ll half = qexp(base, power / 2);
  ll ans = (half * half) % MOD;
  if (power & 1) ans = (ans * base) % MOD;
  return ans;
}

int main() {
  ll m, n;
  cin >> m >> n;
  cout << (qexp(m, n) - m * qexp(m - 1, n - 1) % MOD + MOD) % MOD << endl;
}
```

## X-Factor Chains

```cpp
/**
 * @file poj/3421/main.cpp
 * @author Ruiming Guo (guoruiming@stu.scu.edu.cn)
 * @brief
 * 输入正整数x，求 x
 * 的因子组成的满足任意前一项都能整除后一项的序列的最大长度，
 * 以及所有不同序列的个数
 *
 *
 * @version 0.1
 * @date 2022-04-29
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <iostream>
using namespace std;
const int N = 22;
typedef unsigned long long ull;
ull factorial[N + 1] = {1};

// 阶乘数组 打表
void init(int n) {
  for (int i = 1; i <= n; ++i) factorial[i] = factorial[i - 1] * i;
}

void solve(ull x) {
  ull fcount = 0,       // x 的因子的数量
      denominator = 1;  // 分母
  for (ull i = 2; i <= x / i; ++i) {
    if (x % i == 0) {  // i 是 x 的一个因子
      int ecount = 0;  // 求这个因子的次方数
      while (x % i == 0) {
        ecount++;
        x /= i;
      }
      // i, i^2, i^3, ..., i^ecount 都是 x 的因子
      fcount += ecount;
      denominator *= factorial[ecount];
    }
  }
  // 如果 x 不是 1，需要单独统计一次因子 1
  // 是 1 的话，fcount应该是 0
  if (x != 1) fcount += 1;

  cout << fcount << " "                    // 最大长度
       << factorial[fcount] / denominator  // 组合计数，因子个数
       << endl;
}
int main() {
  init(N);
  ull x;
  while (cin >> x) solve(x);
  return 0;
}
```

## 排列计数

```cpp
/**
 * @file luogu/4071/main.cpp
 * @author Ruiming Guo (guoruiming@stu.scu.edu.cn)
 * @brief 错位排列问题，f[i] = (i-1)*f[i-1] + (i-1)*f[i-2]
 * @version 0.1
 * @date 2022-04-29
 * @note SDOI 卡常数数正常现象
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
ll mod = 1e9 + 7;
const int N = 1000010;
ll factorial[N];
ll f[N];
ll qexp(ll a, ll b) {
  ll res = 1;
  while (b) {
    if (b & 1) res *= a, res %= mod;
    a = a * a % mod;
    b >>= 1;
  }
  return res;
}
inline ll inv(ll x) { return qexp(x, mod - 2); }

void init() {
  factorial[0] = 1;
  for (ll i = 1; i < N; ++i) factorial[i] = factorial[i - 1] * i % mod;
  f[0] = 1, f[1] = 0, f[2] = 1;
  for (ll i = 3; i < N; ++i)
    f[i] = ((i - 1) * f[i - 1] % mod + (i - 1) * f[i - 2] % mod) % mod;
}
inline ll C(ll x, ll y) {
  return factorial[x] * inv(factorial[x - y]) % mod * inv(factorial[y]) % mod;
}
int main() {
  ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
  init();
  ll tc;
  cin >> tc;
  while (tc--) {
    ll n, m;
    cin >> n >> m;
    ll ans = C(n, m) * f[n - m] % mod;
    cout << ans << endl;
  }
}
```

## Marks Distribution

```cpp
/**
 * @file uva/10910/main.cpp
 * @author Ruiming Guo (guoruiming@stu.scu.edu.cn)
 * @brief 一个学生选修了 N 门课程，
 * 一共得到了 T 分
 * 已知他都及格了
 * 给出各门课程的及格分数线
 * 问该学生可能的成绩组合的种类数
 * @version 0.1
 * @date 2022-04-29
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
  int tc;
  cin >> tc;
  while (tc--) {
    ll x, y, z;
    cin >> x >> y >> z;
    y -= x * z;
    ll ans = 0;
    if (y >= 0) {
      ans = 1;
      ll now = y + x - 1 - max(x - 1, y);
      ll tot = 1;
      for (int i = 1; i <= now; ++i) {
        tot *= i;
        ans *= (y + x - i);
        ll hel = __gcd(tot, ans);
        tot /= hel;
        ans /= hel;
      }
    }
    cout << ans << endl;
  }
}
```

## Neon Sign

```cpp
/**
 * @file uva/1510/main.cpp
 * @author Ruiming Guo (guoruiming@stu.scu.edu.cn)
 * @brief 给定一个有 n 个顶点的完全图，给定图上各边为红色或蓝色（Beat Saber!）
 * 问同种颜色的边构成的三角形的数量
 *
 * 一共构成的三角形C(n, 3) = n*(n-1)*(n-2)/6
 *
 * 对一个“不纯”的三角形，一定是其中两条同色一条异色，
 * 那么对其三个点，有两个点会出现以每个点为端点的两条边异色的情况，
 * 所以只需要统计这种情况（每个点引出的红边数量*该点引出蓝边数量）
 * 记录为 sum，
 * “不纯”的三角形的个数即为 sum / 2
 *
 * @version 0.1
 * @date 2022-04-29
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <bits/stdc++.h>
using namespace std;
const int N = 1010;
int n, sum, t[N][2];
int main() {
  int tc;
  cin >> tc;
  while (tc--) {
    memset(t, 0, sizeof(t));
    cin >> n;
    sum = 0;
    for (int i = 1; i <= n - 1; ++i) {
      for (int j = i + 1; j <= n; ++j) {
        int color;
        cin >> color;
        ++t[i][color];
        ++t[j][color];
      }
    }
    for (int i = 1; i <= n; ++i) sum += t[i][0] * t[i][1];
    printf("%lld\n", 1LL * n * (n - 1) * (n - 2) / 6 - sum / 2);
  }
}
```

## Cheerleaders

```cpp
/**
 * @file uva/11806/main.cpp
 * @author Ruiming Guo (guoruiming@stu.scu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-04-29
 * 在一个 n×m 的矩阵中摆放 k 个石子，要求第一行、第一列、第 m 行、第
 * n 列必须有石子，求方案总数。T≤50,n≤20,m≤20,k≤500。
 *
 *
 * @copyright Copyright (c) 2022
 *
 *
 */

#include <bits/stdc++.h>
using namespace std;
#define int long long
const int N = 510, mod = 1e6 + 7;
int T, t, m, n, k, c[N][N], ans;
// 递推法计算组合数
void init() {
  c[0][0] = 1;
  for (int i = 1; i <= 500; ++i) {
    c[i][0] = 1;
    for (int j = 1; j <= i; ++j)
      c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % mod;
  }
}

signed main() {
  scanf("%lld", &T);
  init();
  while (T--) {
    scanf("%lld%lld%lld", &m, &n, &k);
    ans = 0;  // 要往下减多少种情况
    // 条件：第一行有石子、第一列有石子、第 m 行有石子、第 n 列有石子
    // 枚举状态，第 i 位为 1 表示 i 条件不被满足
    for (int S = 0; S < (1 << 4); ++S) {
      int x = m,                   // 可以放棋子的行数
          y = n,                   // 可以放棋子的列数
          cnt = 0;                 // 当前状态下有cnt个条件不被满足
      for (int i = 0; i < 4; i++)  // 枚举条件
        if ((S >> i) & 1) {        // 第i个条件未被满足
          if (i & 1)  //若是条件0或条件2，那么有一行不能放（即x--）
            x--;
          else  //若是条件1或条件3，那么有一列不能放（即y--）
            y--;
          cnt++;  // 未满足的条件数+1
        }
      if (!cnt) continue;  // 都满足了，不往下减
      if (cnt & 1)
        ans = (ans + c[x * y][k] % mod) % mod;
      else
        ans = (ans + mod - c[x * y][k] % mod) % mod;
    }
    // 总方案数减有任意一个条件不被满足的方案数
    printf("Case %lld: %lld\n", ++t, (c[n * m][k] % mod - ans + mod) % mod);
  }
}
```

## Triangle Counting

```cpp
/**
 * @file uva/11401/main.cpp
 * @author Ruiming Guo (guoruiming@stu.scu.edu.cn)
 * @brief 给定 n 条边，长度分别为1, 2, 3, ..., n
 * 。用其中三条边构成一个三角形，一条边只能使用一次，有多少种不同的方案？
 * @version 0.1
 * @date 2022-04-29
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <bits/stdc++.h>
using namespace std;
const int N = 1000010;
long long f[N];
int main() {
  for (long long i = 4; i < N; i++) {
    f[i] = f[i - 1] + ((i - 1) * (i - 2) / 2 - (i - 1) / 2) / 2;
  }
  long long n;
  while (cin >> n) {
    if (n < 3) break;
    cout << f[n] << endl;
  }
  return 0;
}
```

## Irrelevant Elements

```cpp
#include <cmath>
#include <cstdio>
#include <cstring>

int m, n, a[1000010], p[1000010], now[1000010], ans[1000010];
int main() {
  int i, j, k, x, y, z, tot = 0, cnt = 0;
  bool flag;
  scanf("%d%d", &n, &m);
  x = sqrt(m + 0.5);
  for (i = 2; i <= x; i++)
    if (m % i == 0) {
      a[++tot] = i;
      while (m % i == 0) {
        m /= i;
        p[tot]++;
      }
    }
  if (m > 1) {
    a[++tot] = m;
    p[tot] = 1;
  }
  for (i = 1; i < n - 1; i++) {
    x = n - i;
    y = i;
    for (j = 1; j <= tot; j++)
      while (x % a[j] == 0) {
        x /= a[j];
        now[j]++;
      }
    for (j = 1; j <= tot; j++)
      while (y % a[j] == 0) {
        y /= a[j];
        now[j]--;
      }
    flag = 1;
    for (j = 1; j <= tot; j++)
      if (p[j] > now[j]) {
        flag = 0;
        break;
      }
    if (flag) ans[++cnt] = i + 1;
  }
  printf("%d\n", cnt);
  for (i = 1; i <= cnt; i++) printf("%d%c", ans[i], i == cnt ? '\n' : ' ');
}
```
