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
