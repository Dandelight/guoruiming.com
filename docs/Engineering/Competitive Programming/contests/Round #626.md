## [B. Counting Subrectangles](https://codeforces.com/contest/1323/problem/B)

给定$a_i$和$b_i$两个由$0$和$1$组成的数组，定义矩阵$c_{ij}=a_i\times b_j$，计算矩阵$C$中由$1$组成的面积为$k$的矩形个数。

假设有数组$x, y$满足$x_i \times y_i = k$，可见问题的关键是找到$a$中有多少个连续$x_i$个$1$，$b$中有多少个连续$y_i$个$1$。

于是我们需要使用一个$cnt$数组，$cnt_i$记录$a$中有多少个连续$i$个$1$；$b$数组同理。建立$cnt$数组需扫描$a$数组，如果遇到连续的$\ell$个$1$，就将$cnt_i$增加$\ell - i + 1$。时间复杂度$O(n)$(?)

## [D. Present](https://codeforces.com/contest/1323/problem/D)

有一个数组，计算

$$
(a_1 + a_2) \oplus (a_1 + a_3) \oplus \ldots \oplus (a_1 + a_n) \\ \oplus (a_2 + a_3) \oplus \ldots \oplus (a_2 + a_n) \\ \ldots \\ \oplus (a_{n-1} + a_n) \\
$$

每一位来考虑，可以发现第$k$位数字只与$a_i$的第$0\sim k$位有关，也就是说和$a_i \mod 2^{k+1}$有关。取模之后$a_i$之和不会超过$2^{k+2}-2$，第$k$位为$1$当且仅当有$(a_i, a_j)$之和落在$[2^k, 2^{k+1})$或$[2^{k+1}+2^k, 2^{k+2}-2]$。

```cpp
// https://codeforces.com/contest/1322/submission/72634815
#include <bits/stdc++.h>

using namespace std;

typedef long long ll;

int sum[1<<25],num[400005];

int main() {
  int n;
  scanf("%d",&n);
  for(int i=1;i<=n;i++) scanf("%d",&num[i]);
  int ans=0;
  for(int len=2;len<=(1<<25);len<<=1) {
  	for(int i=0;i<len;i++) sum[i]=0;
  	int u=len>>1,v=len-1;
  	for(int i=1;i<=n;i++) sum[num[i]&v]++;
  	for(int i=1;i<len;i++) sum[i]+=sum[i-1];
  	ll s=0;
  	for(int i=1;i<=n;i++) {
  		int t=(num[i]&v);
  		int l=((u-t+len)&v),r=((v-t+len)&v);
  		if (l<=r) s+=sum[r]-sum[l-1];
  		else s+=n-(sum[l-1]-sum[r]);
  		if (((t+t)&v)>=u) s--;
	  }
	s>>=1;
	if (s&1) ans^=u;
  }
  printf("%d\n",ans);
  return 0;
}
```

## [1324D - Pair of Topics](https://codeforces.com/contest/1324/problem/D)

题目说给出$a_i, b_i$两个数组，求使得$a_i+a_j>b_i+b_j, i<j$的$(i, j)$对数。等式可以改写为$(a_i - b_i) + (a_j - b_j) > 0$，定义$c_i = a_i + b_i$，也就是求$i<j$且$c_i+c_j>0$的$(i, j)$对数。

可以先对$c$数组升序排序后枚举$j$，对每个$j$，寻找$i$使得$c_j > 0$且$c_j > -c_i$。$i$可以通过二分`std::lower_bound`找到。每找到一个，答案增加$j-i$。

```cpp
// https://codeforces.com/blog/entry/74714
#include <bits/stdc++.h>

using namespace std;

int main() {
#ifdef _DEBUG
	freopen("input.txt", "r", stdin);
//	freopen("output.txt", "w", stdout);
#endif

	int n;
	cin >> n;
	vector<int> a(n), b(n);
	for (auto &it : a) cin >> it;
	for (auto &it : b) cin >> it;
	vector<int> c(n);
	for (int i = 0; i < n; ++i) {
		c[i] = a[i] - b[i];
	}
	sort(c.begin(), c.end());

	long long ans = 0;
	for (int i = 0; i < n; ++i) {
		if (c[i] <= 0) continue;
		int pos = lower_bound(c.begin(), c.end(), -c[i] + 1) - c.begin();
		ans += i - pos;
	}

	cout << ans << endl;

	return 0;
}
```

## [1324D - Sleeping Schedule](https://codeforces.com/contest/1324/problem/E)

题目大意如下：每天有$h$个小时，Vova 每天睡$n$次，第$i$次睡觉的时间距离上一次苏醒$a_i$小时，每次睡$h$小时。Vova 在$0$时刻苏醒。

定义第$i$次睡觉是**好的**当且仅当该次睡眠开始于$l$小时和$r$小时之间。

Vova 有一定的自制力，在第$i$次睡眠之前他可以**选择**在$a_i$小时后入睡或$a_i-1$小时后入睡。

求**最大的好的睡眠次数**。

这显然是一个动态规划问题~~（虽然我在比赛里没看出来~~，状态为好的睡眠次数，状态转移为按时睡觉或早睡早起，属性为最大值。令$dp_{i, j}$为第$i$次睡眠，其中$j$次早睡时最大“好的”睡眠次数，显而易见答案为$\max\limits_{j=0}^{n} dp_{n, j}$。初始时，$dp_{i, j} = -\infty$，$dp_{0,0}=0$。

每次状态转移有两种选择：按时睡觉与早睡一小时。

- 按时睡觉：$dp_{i+1, j} = \max(dp_{i+1, j}, dp_{i, j}+|(s-j)\mod h \in [l, r]|)$
- 早睡早起：$dp_{i+1, j+1} = \max(dp_{i+1, j+1}, dp_{i, j} + |s - j - 1 \mod h \in [l, r]|)$

Don't forget to don't make transitions from unreachable states.

时间复杂度：$O(n^2)$

```cpp
#include <bits/stdc++.h>

using namespace std;

bool in(int x, int l, int r) {
	return l <= x && x <= r;
}

int main() {
#ifdef _DEBUG
	freopen("input.txt", "r", stdin);
//	freopen("output.txt", "w", stdout);
#endif

	int n, h, l, r;
	cin >> n >> h >> l >> r;
	vector<int> a(n);
	for (auto &it : a) cin >> it;
	vector<vector<int>> dp(n + 1, vector<int>(n + 1, INT_MIN));
	dp[0][0] = 0;
	int sum = 0;
	for (int i = 0; i < n; ++i) {
		sum += a[i];
		for (int j = 0; j <= n; ++j) {
			dp[i + 1][j] = max(dp[i + 1][j], dp[i][j] + in((sum - j) % h, l, r));
			if (j < n) dp[i + 1][j + 1] = max(dp[i + 1][j + 1], dp[i][j] + in((sum - j - 1) % h, l, r));
		}
	}

	cout << *max_element(dp[n].begin(), dp[n].end()) << endl;

	return 0;
}
```

## [1409B - Minimum Product](https://codeforces.com/problemset/problem/1409/B)

给定四个整数$a, b, x, y$，其中$a\ge x$且$b \ge y$，一共可以进行$n$次操作，每次可以将$a$或$b$减一，但必须保证$a\ge x$且$b \ge y$，求最终$a\times b$的最小值。

题目的思路是：当我们开始减一个数，最好的方法是一直减到不能再减为止。

先减哪个好呢？小孩子才做选择。

```cpp
#include <bits/stdc++.h>
using namespace std;
int main() {
	int t;
	cin >> t;
	while (t--) {
		int a, b, x, y, n;
		cin >> a >> b >> x >> y >> n;
		long long ans = 1e18;
		for (int i = 0; i < 2; ++i) {
			int da = min(n, a - x);
			int db = min(n - da, b - y);
			ans = min(ans, (a - da) * 1ll * (b - db));
			swap(a, b);
			swap(x, y);
		}
		cout << ans << endl;
	}
	return 0;
}
```

## [1404A - Balanced Bitstring](https://codeforces.com/problemset/problem/1404/A)

bitstring 是仅包含 0 和 1 的字符串。给定一个包含`0`、`1`和`?`的字符串$s$和一个正整数$k$，求是否可以对$s$中的`?`赋值使其中每连续的$k$个字符都是一个 bitstring。

可以发现，$s_i = s_{i+k}$，由此推导可知$t_j=t_j\quad \text{if} \quad i \equiv j\pmod k$。因此，当我们可以扫描整个字符串，检查是否有$t_i = t_{i \mod k}$。如果$t_i$不是`?`而$t_{i\mod k}$是`?`，那么将$t_{i \mod k}$赋值为$t_i$。扫描完成后再检查一边$t_0, \ldots, t_{k-1}$中`1`和`0`的数量是否超过$k/2$，如果超过说明为假。

```cpp
#include <bits/stdc++.h>
using namespace std;
int n, k, t;
string s;
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cin >> t;
    while (t--) {
        cin >> n >> k >> s;
        int zer = 0, one = 0;
        bool chk = true;
        for (int i = 0; i < k; i++) {
            int tmp = -1;
            for (int j = i; j < n; j += k) {
                if (s[j] != '?') {
                    if (tmp != -1 && s[j] - '0' != tmp) {
                        chk = false;
                        break;
                    }
                    tmp = s[j] - '0';
                }
            }
            if (tmp != -1) {
                (tmp == 0 ? zer : one)++;
            }
        }
        if (max(zer, one) > k / 2) {
            chk = false;
        }
        cout << (chk ? "YES\n" : "NO\n");
    }
}
```

## [1405D - Tree Tag](https://codeforces.com/contest/1405/problem/D)

Alice 和 Bob 分别站在一棵树的两个结点$a, b$上，Alice 每次可以走过$da$条边，Bob 可以走过$db$条边。Alice 先手，二人交替操作，经过足够多的步数后，当 Alice 和 Bob 落到同一个节点上 Alice 胜，否则 Bob 胜。另外已知 Bob 和 Alice 都是绝顶聪明的人，都会执行最佳操作，问谁最终获胜。

这道题明显是一道博弈论问题，并且需要**分类讨论**。

### 1. $\mathrm{dist}(a, b) \le da$

Alice 交出一个闪现大步一跃一把抓住 Bob，Bob，卒。

### 2. $2da > 树的直径$

Alice 首先到达树的重心，之后 Bob 插翅难逃。

### 3. $db \le 2da$

设 Alice 所在的点$a$为树根，$b$点所在的子树有$k$个结点。Alice 向$b$一定一步，$k$必然减小，而 Bob 不能逃到其他子树，因为一定会被捉。所以 Bob 只能眼睁睁看着 Alice 步步紧逼。Bob，卒。

### 4. $db > 2da$

此时 Bob 终于有了回旋的余地，机智的 Bob 将采取以下的方案。首先因为不在情况 1 中，Bob 躲过第一招；又因为不在情况 2 中，树上至少有一个点与 Alice 的距离大于$da$，也就是 Alice 一步走不到的。如果 Bob 在一个 Alice 下一步走不到的点，那么他可以待在那里；否则，设这个点为$v$，$\mathrm{dist}(a, v) = da + 1$，那么$\mathrm{dist}(b, v) \le \mathrm{dist}(b, a) + \mathrm{dist}(a, v) \le da +(da+1)\le 2da+1 \le db$，写了这么长就是为了证明 Bob 可以一步踏入安全区。

容易证明以上的分类讨论是没有遗漏的。至于怎么想到，也许需要一点常识？

找直径的算法就是指定任意一个节点$u$为根，选出以$u$为端点的最短路径和次短路径长度，求和。可以证明以任意一个点开始`dfs`搜到的直径的数值是相同的。

时间复杂度$O(n)$。

```cpp
#include <bits/stdc++.h>

using namespace std;

const int N = 1e5 + 5;
int n, a, b, da, db, depth[N];
vector<int> adj[N];
int diam = 0;

int dfs(int x, int p) {
    int len = 0;
    for(int y : adj[x]) {
        if(y != p) {
            depth[y] = depth[x] + 1;
            int cur = 1 + dfs(y, x);
            diam = max(diam, cur + len);
            len = max(len, cur);
        }
    }
    return len;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    int te;
    cin >> te;
    while(te--) {
        cin >> n >> a >> b >> da >> db;
        for(int i = 1; i <= n; i++) adj[i].clear();
        for(int i = 0; i < n - 1; i++) {
            int u, v;
            cin >> u >> v;
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
        diam = 0;
        depth[a] = 0;
        dfs(a, -1);
        cout << (2 * da >= min(diam, db) || depth[b] <= da ? "Alice" : "Bob") << '\n';
    }
}
```

## [1404C - Fixed Point Removal](https://codeforces.com/problemset/problem/1404/C)

这是一道 Div. 1 的 C 题，先放在这里吧。
