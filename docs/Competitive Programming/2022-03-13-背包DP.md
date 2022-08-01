**20220313 背包 DP**

|                |                                          |       |                                             |
| -------------- | ---------------------------------------- | ----- | ------------------------------------------- |
| 01 背包转化    | USACO 2015 Dec G. Fruit Feast            | 提高- | https://www.luogu.com.cn/problem/P4817      |
| 01 背包转化    | CF837D Round Subset                      | 提高- | http://codeforces.com/contest/837/problem/D |
| 01 背包转化    | CEOI2018 Cloud Computing                 | 提高  | https://www.luogu.com.cn/problem/P6359      |
| 完全背包方案数 | [NOIP2018 提高组] 货币系统               | 提高  | https://www.luogu.com.cn/problem/P5020      |
|                |                                          |       |                                             |
| 01 背包转化    | USACO 2020 Open G. Exercise              | 提高+ | https://www.luogu.com.cn/problem/P6280      |
| 完全背包方案数 | USACO 2019 Jan G. Cow Poetry             | 提高+ | https://www.luogu.com.cn/problem/P5196      |
| 01 背包转化    | POI2004 - Maximal Orders of Permutations | NOI-  | https://www.luogu.com.cn/problem/P5919      |
| 分数规划       | USACO 2018 Open G. Talent Show           | 提高+ | https://www.luogu.com.cn/problem/P4377      |
| 分组背包       | [BJOI2019] 排兵布阵                      | 提高+ | https://www.luogu.com.cn/problem/P5322      |

## Fruit Feast

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 5'000'000 + 10;
int a, b, t;
bool f[N] = {1};
int main() {
  cin >> t >> a >> b;
  // 吃柠檬
  for (int i = a; i <= t; ++i) f[i] |= f[i - a];
  // 吃橘子
  for (int i = b; i <= t; ++i) f[i] |= f[i - b];
  // 喝水，at most one time
  for (int i = 1; i <= t; ++i) f[i >> 1] |= f[i];
  // 继续吃柠檬
  for (int i = a; i <= t; ++i) f[i] |= f[i - a];
  // 继续吃橘子
  for (int i = b; i <= t; ++i) f[i] |= f[i - b];
  int ans = t;
  while (!f[ans]) ans--;
  cout << ans << endl;
}
```

## Subset

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 222;
const int F = 30;
int dp1[N][N * F];
int dp2[N][N * F];

void upd(int& a, int b) { a = max(a, b); }

int main() {
  int n, k;
  scanf("%d%d", &n, &k);
  vector<int> d2(n);
  vector<int> d5(n);
  for (int i = 0; i < n; i++) {
    ll x;
    scanf("%lld", &x);
    for (; x % 5 == 0; x /= 5) {
      d5[i]++;
    }
    for (; x % 2 == 0; x /= 2) {
      d2[i]++;
    }
  }
  memset(dp1, -63, sizeof(dp1));
  dp1[0][0] = 0;
  for (int i = 0; i < n; i++) {
    memset(dp2, -63, sizeof(dp2));
    for (int j = 0; j <= k; j++) {
      for (int t = 0; t <= j * F; t++) {
        upd(dp2[j][t], dp1[j][t]);
        upd(dp2[j + 1][t + d5[i]], dp1[j][t] + d2[i]);
      }
    }
    memcpy(dp1, dp2, sizeof(dp2));
  }
  int answer = 0;
  for (int i = 0; i < k * F; i++) answer = max(answer, min(i, dp1[k][i]));

  cout << answer << endl;
  return 0;
}
```

## Cloud Computing

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 4e3 + 10, M = 1e5 + 10;
int n, m, cnt;
typedef long long ll;
ll dp[M], ans;
struct P {
  int c, f, v;
  bool operator<(const P &a) const {
    // 时钟频率相同时，价格低的的优先；
    // 买入的价值为负，故买入在前
    if (f == a.f) return v < a.v;
    // 时钟频率大的优先
    return f > a.f;
  }
} a[N];
int main() {
  cin >> n;
  for (int i = 1; i <= n; ++i) {
    cin >> a[i].c >> a[i].f >> a[i].v;
    a[i].v = -a[i].v;
  }
  cin >> m;
  for (int i = n + 1; i <= n + m; ++i) {
    cin >> a[i].c >> a[i].f >> a[i].v;
  }
  sort(a + 1, a + n + m + 1);
  memset(dp, -0x3f, sizeof dp);  // 初始化为-INF
  dp[0] = 0;
  for (int i = 1; i <= n + m; ++i) {
    auto [c, f, v] = a[i];
    if (v < 0) {  // 买入
      for (int j = cnt; j >= 0; --j) {
        dp[j + c] = max(dp[j + c], dp[j] + v);
      }
      cnt += c;  // 记录当前可能的最大核心数
    } else {     // 卖出
      for (int j = 0; j <= cnt - c; ++j) {
        dp[j] = max(dp[j], dp[j + c] + v);
      }
    }
  }
  for (int i = 0; i <= cnt; ++i) ans = max(ans, dp[i]);
  cout << ans << endl;
}
```

## 货币系统

```cpp
#include<iostream>
#include<cstring>
#include<algorithm>
using namespace std;
const int maxn = 100+6;
const int maxa = 25000+7;
int a[maxa], dp[maxa];
int n;
int main() {
  int t; cin>>t;
  while(t--) {
    cin>>n; for(int i=0; i<n; ++i) cin>>a[i];
    memset(dp, 0, sizeof(dp));
    sort(a, a+n);
    int ans = n;
    dp[0] = 1; // EXTREMELY IMPORTANT
    for(int i=0; i<n; ++i) {
      if(dp[a[i]]) {
        ans--;
      }
      else {
        for(int j=a[i]; j<maxa; j++) {
          dp[j] = dp[j] | dp[j - a[i]];
        }
      }
    }
    cout<<ans<<'\n';
  }
}
```

## Exercise

```cpp
#include <bits/stdc++.h>
#define int long long
#define ffor(i, a, b) for (int i = (a); i <= (b); i++)
#define roff(i, a, b) for (int i = (a); i >= (b); i--)
using namespace std;
const int MAXN = 1e4 + 10, MAXM = 1230;
int n, m, dp[MAXM][MAXN], vis[MAXN];
vector<int> pr;
void init(int MAX) {
  pr.push_back(0);
  ffor(i, 2, MAX) {
    if (!vis[i]) pr.push_back(i);
    for (int j = 1; j < pr.size() && pr[j] * i <= MAX; j++) {
      vis[i * pr[j]] = 1;
      if (i % pr[j] == 0) break;
    }
  }
  return;
}
signed main() {
  scanf("%lld %lld", &n, &m);
  init(n);
  dp[0][0] = 1;
  ffor(i, 1, pr.size() - 1) {
    int base = pr[i];
    ffor(j, 0, n) dp[i][j] = dp[i - 1][j];
    while (base <= n) {
      ffor(j, base, n) dp[i][j] += dp[i - 1][j - base] * base, dp[i][j] %= m;
      base *= pr[i];
    }
  }
  int ans = 0;
  ffor(i, 0, n) ans += dp[pr.size() - 1][i], ans %= m;
  printf("%lld", ans);
  return 0;
}
```

## Cow poetry

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 5010, mod = 1'000'000'007;
int n, m, k;
int f[N][N], s[N], l[N], y[N], cnt[30];
inline int qexp(int x, int k) {
  int s = 1;
  while (k) {
    if (k & 1) s = (1LL * s * x) % mod;
    x = (1LL * x * x) % mod;
    k >>= 1;
  }
  return s;
}
int main() {
  cin >> n >> m >> k;
  s[0] = 1;
  for (int i = 1; i <= n; ++i) cin >> l[i] >> y[i];
  for (int i = 1; i <= k; ++i) {
    for (int j = 1; j <= n; ++j) {
      if (i >= l[j]) {
        f[i][y[j]] = (f[i][y[j]] + s[i - l[j]]) % mod;
        s[i] = (s[i] + s[i - l[j]]) % mod;
      }
    }
  }
  for (int i = 1; i <= m; ++i) {
    char ch;
    cin >> ch;
    cnt[ch - 'A']++;
  }
  int res = 1;
  for (int i = 0; i < 26; ++i) {
    if (!cnt[i]) continue;
    int ans = 0;
    for (int j = 1; j <= n; ++j) {
      if (f[k][j]) ans = (ans + qexp(f[k][j], cnt[i])) % mod;
    }
    res = 1LL * res * ans % mod;
  }
  cout << res << endl;
}
```

## Maximal Order of Permutations

```cpp
#include <bits/stdc++.h>
using namespace std;

const int MAX_N = 1e4, MAX_P = 70;

bool composite[MAX_N + 1];
vector<int> primes = { 2,   3,   5,   7,   11,  13,  17,  19,  23,  29,
					   31,  37,  41,  43,  47,  53,  59,  61,  67,  71,
					   73,  79,  83,  89,  97,  101, 103, 107, 109, 113,
					   127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
					   179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
					   233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
					   283, 293, 307, 311, 313, 317, 331, 337, 347, 349
					};

double mem[MAX_N + 1][MAX_P];
bool visited[MAX_N + 1][MAX_P];
int backtrack[MAX_N + 1][MAX_P];

double dp(int n, int pos = 0) {
	if (pos == primes.size() || primes[pos] > n) return 0;
	if (visited[n][pos]) return mem[n][pos];

	double ans = dp(n, pos + 1);
	backtrack[n][pos] = 0;
	for (int p_pow = primes[pos], expo = 1; p_pow <= n; p_pow *= primes[pos], expo++) {
		double potential = expo * log(primes[pos]) + dp(n - p_pow, pos + 1);
		if (ans < potential) {
			ans = potential;
			backtrack[n][pos] = p_pow;
		}
	}
	visited[n][pos] = true;
	return mem[n][pos] = ans;
}

int main() {
	int n;
	scanf("%d", &n);
	while (n--) {
		int m;
		scanf("%d", &m);
		dp(m);
		vector<int> lens;
		for (int pos = 0; pos < primes.size(); m -= backtrack[m][pos++]) {
			if (backtrack[m][pos]) lens.push_back(backtrack[m][pos]);
		}
		while (m--) lens.push_back(1);
		sort(lens.begin(), lens.end());
		for (int i = 0, j = 1; i < lens.size(); j += lens[i++]) {
			for (int k = 1; k < lens[i]; k++) printf("%d ", j + k);
			printf("%d ", j);
		}
		printf("\n");
	}
	return 0;
}
```

## Talent Show

```cpp
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
int n, W;
int w[300], t[300];
long long f[10000];
bool check(int z) {
  memset(f, 128, sizeof(f));
  f[0] = 0;
  long long tmp = f[W];
  for (int i = 1; i <= n; i++) {
    for (int j = W; j >= 0; j--)
      if (f[j] != tmp) {
        int jj = j + w[i];
        jj = min(jj, W);
        f[jj] = max(f[jj], f[j] + t[i] - (long long)w[i] * z);
      }
  }
  return f[W] >= 0;
}
int bisect() {
  int l = 0, r = 1000000;
  while (l <= r) {
    int mid = l + r >> 1;
    if (check(mid))
      l = mid + 1;
    else
      r = mid - 1;
  }
  return l - 1;
}
int main() {
  scanf("%d%d", &n, &W);
  for (int i = 1; i <= n; i++) {
    scanf("%d%d", &w[i], &t[i]);
    t[i] *= 1000;
  }
  printf("%d", bisect());
  return 0;
}
```

## 排兵布阵

设$dp[i][j]$表示第$i$个城堡时，已派出$j$个士兵。决策时，贪心派出恰好严格大于某一玩家派出的数量的两倍（不然浪费）。我们发现又可以排序预处理$a[i][j]$出表示第$i$个城堡，出兵数量第$j$大的人出兵数量（因为这样可以很容易算出贡献，即为$k \times i$

dp 转移方程即为：

$$
dp[j] = \max(dp[j-a[i][k] \times2-1]+k\times i, dp[j])
$$

```cpp
#include <cstdio>
#include <algorithm>
#define MAX(A,B) ((A)>(B)?(A):(B))
using namespace std;
int s,n,m,dp[20002],a[110][110],ans;
signed main(){
    scanf("%d %d %d", &s, &n, &m);
    for(int i=1;i<=s;++i)
        for(int j=1;j<=n;++j)
            scanf("%d", &a[j][i]);
    for(int i=1;i<=n;++i)
        sort(a[i]+1, a[i]+1+s);
    for(int i=1;i<=n;++i)
        for(int j=m;j>=0;--j) //倒序枚举已派出兵
            for(int k=1;k<=s;++k) //对s个玩家决策
                if(j>a[i][k]*2)
                    dp[j]=MAX(dp[j-a[i][k]*2-1]+k*i, dp[j]);
    for(int i=0;i<=m;++i) ans=MAX(ans, dp[i]);
    printf("%d\n", ans);
    return 0;
}
```
