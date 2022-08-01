| 1h   | AtCoder DP Contest - Matching   | 提高  | https://www.luogu.com.cn/problem/AT4536       |
| ---- | ------------------------------- | ----- | --------------------------------------------- |
| 1.5h | CF1316E Team Building           | 提高+ | https://codeforces.com/contest/1316/problem/E |
| 1.5h | USACO 2014 Dec G. Guard Mark    | 提高+ | https://www.luogu.com.cn/problem/P3112        |
| 1h   | Headmaster's Headache, UVa10817 | 提高+ | https://www.luogu.com.cn/problem/UVA10817     |

| 1h   | [USACO13NOV G]No Change            | 提高+ |      | https://www.luogu.com.cn/problem/P3092           |
| ---- | ---------------------------------- | ----- | ---- | ------------------------------------------------ |
| 1h   | Hacker's Crackdown, UVa11825       | 提高+ |      | https://www.luogu.com.cn/problem/UVA11825        |
| 1.5h | Sharing Chocolate, WF2010, UVa1099 | NOI-  |      | https://www.luogu.com.cn/problem/UVA1099         |
| 2h   | COCI2016 - Burza                   | NOI-  | 博弈 | https://www.luogu.com.cn/problem/P6499           |
| 2h   | CF1043F Make It One                | NOI-  | GCD  | https://codeforces.com/problemset/problem/1043/F |

## Matching

给定二分图，两个集合都有 $1 \le N \le 21$ 个点，$a_{i,j}=1$ 表示第一个集合第 $i$ 个点与第二个集合第 $j$ 个点连边，$a_{i, j}=1$表示不连边。求二分图完备匹配数，答案对 $10^9+7$ 取模。

思路: 状压 DP

**状态**：设$dp[i][j]$表示已经考虑了节点 $\{0,1,\ldots,i \}$,右侧所用的点表示为$mask$.

**转移**：对于$dp[i][j]$,枚举节点$i$的所有邻居$j$,如果$mask$中含有$j$的话,方案数就加上$dp[i-1][mask-(i \ll j)]$.

更进一步，$\{0,1,\ldots,i \}$一共有$i+1$个节点，所以只有$mask$中一的数量等于$i+1$的时候状态才会有意义。于是我们只需要保留$mask$这一堆，另一维可以由$\text{\_\_builtin\_popcount}(mask)-1$计算得出。

```cpp
/**
 * @file Matching
 * @author Ruiming Guo (guoruiming@stu.scu.edu.cn)
 * @brief 又到了乱点鸳鸯谱的时间。
 *
 * 给定二分图，两个集合都有 N 个点，a_{i,j}=1 表示第一个集合第 i
 * 个点与第二个集合第 j 个点连边。 求二分图完备匹配数，答案对 10^9+7 取模
 * @version 1.0
 * @date 2022-05-11
 *
 * @copyright Copyright (c) 2022
 *
 **/

#include <bits/stdc++.h>
using namespace std;
const int mod = 1e9 + 7;
const int N = 21;
int n, a[N][N], dp[1 << N];
int main() {
  cin >> n;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) cin >> a[i][j];

  dp[0] = 1;
  for (int mask = 1; mask < 1 << n; ++mask) {
    int i = __builtin_popcount(mask) - 1;
    for (int j = 0; j < n; ++j)
      if (a[i][j] && mask & 1 << j) {
        dp[mask] = (dp[mask] + dp[mask - (1 << j)]) % mod;
      }
  }
  cout << dp[(1 << n) - 1] << endl;
}
```

## Team Building

从$n$位候选人中选出$p$位球员和$k$位观众。对于第$i$个人，

- 选作观众能使团队战力增加$a_i$
- 选作第$j$位球员能使团队战力增加$s_{i,j}$

但不能同时作为球员和观众。

问团队能达到的最大战力。

### 解答

首先降序排列$a_i$。

**状态**：$dp[i][mask]$，表示考虑到了$i$名候选人，队员选中情况为$mask$时战力最大值。

如果第$i$名候选被选中踢$j$位置，有转移

$$
dp[i][mask] = dp[i-1][mask \oplus 2^j] + s_{i, j}
$$

如果这名候选人没有选为球员，她一定被选为观众，因为我们已经对$a_i$降序排列了，所以选前面的人当观众一定优于选后边的人当观众。

那么我们需要一个变量$c$记录选到$i$时没有选为球员的人数，也就是已经选择的观众的人数。

所以转移方程的第二部分：

$$
dp[i][mask] =
\begin{cases}
dp[i-1][mask] + a_i &\text{if}\ c < k\\
dp[i-1][mask] & \text{else}
\end{cases}
$$

**时间复杂度**：$O(n\cdot p \cdot 2^p)$

```cpp
// https://codeforces.com/contest/1316/submission/156836529
#include <bits/stdc++.h>
using namespace std;
#define int long long
const int M = 1e5 + 5;
int dp[M][(1 << 7) + 1], skill[M][7];
int ind[M], a[M];

signed main() {
  ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
  int n, p, k;
  memset(dp, -1, sizeof(dp));
  cin >> n >> p >> k;
  for (int i = 1; i <= n; ++i) cin >> a[i];
  for (int i = 1; i <= n; ++i)
    for (int j = 0; j < p; ++j) cin >> skill[i][j];
  for (int i = 1; i <= n; ++i) ind[i] = i;

  sort(ind + 1, ind + n + 1, [](int p, int q) { return a[p] > a[q]; });

  dp[0][0] = 0;
  for (int i = 1; i <= n; ++i) {
    int x = ind[i];
    for (int mask = 0; mask < (1 << p); ++mask) {
      // 1. 选为第 p 位球员的情况
      for (int j = 0; j < p; ++j)
        if (mask & (1 << j) && dp[i - 1][mask ^ (1 << j)] != -1)
          if (int z = dp[i - 1][mask ^ (1 << j)] + skill[x][j]; z > dp[i][mask])
            dp[i][mask] = z;

      // 已经选了 z 名观众可用该方法计算
      int z = (i - 1) - __builtin_popcountll(mask);
      // 2. 选为观众的情况
      if (z < k) {
        // 观众未满将这位选为观众
        if (dp[i - 1][mask] != -1)
          dp[i][mask] = max(dp[i][mask], dp[i - 1][mask] + a[x]);
      } else {
        // 观众已满，不能将这位选为观众，但要更新DP数组
        if (dp[i - 1][mask] != -1)
          dp[i][mask] = max(dp[i][mask], dp[i - 1][mask]);
      }
    }
  }
  cout << dp[n][(1 << p) - 1] << "\n";
  return 0;
}
```

## Guard Mark

FJ 将飞盘抛向身高为$1 \le H \le 1,000,000,000$)的 Mark，但是 Mark 被$2 \le N \le 20$牛包围。牛们可以叠成一个牛塔，如果叠好后的高度大于或者等于 Mark 的高度，那牛们将抢到飞盘。

每头牛都**身高**，**体重**和**耐力**值三个指标。耐力指的是一头牛最大能承受的叠在他上方的牛的重量和。请计算牛们是否能够抢到飞盘。若是可以，请计算牛塔的最大稳定强度。稳定强度是指，在每头牛的耐力都可以承受的前提下，还能够在牛塔最上方添加的最大重量。

```cpp
// https://www.luogu.com.cn/record/75555838
#include <bits/stdc++.h>
using namespace std;
#define int long long
const int N = 21;
int n, H;

int h[N],  // 身高
    w[N],  // 体重
    s[N];  // 耐力
const int INF = 1e18;
int dp[1 << N],  // dp[S] 定义为在 S 状态下剩余的最大耐力值（最大还能扛多少重量
    ans;
signed main() {
  scanf("%lld%lld", &n, &H);
  for (int i = 0; i < n; ++i) scanf("%lld%lld%lld", &h[i], &w[i], &s[i]);

  ans = -INF;
  dp[0] = INF;                          // 当没有
  for (int S = 1; S < (1 << n); ++S) {  // 枚举状态
    dp[S] = -INF;                       // 初始化DP数组为-INF
    int now = 0;                        // 当前高度
    for (int i = 0; i < n; ++i) {
      if (S & (1 << i)) {
        dp[S] = max(dp[S],                        // 原状态
                    min(                          // 转移
                        dp[S ^ (1 << i)] - w[i],  // 假设让这头牛站在最上边
                        s[i]));  // 第 i 头牛自己能扛多少
        now += h[i];             // 牛塔又高了一截
      }
    }
    if (now >= H        // 身高达成
        && dp[S] >= 0)  // 牛还没压死（bushi
      ans = max(dp[S], ans);
  }
  if (ans < 0)
    puts("Mark is too tall");
  else
    printf("%lld\n", ans);
}
```

## Headmaster’s Headache

看不懂了……

```cpp
#include <bits/stdc++.h>

using namespace std;

const int N = 110;
const int MAXN = (1 << 8);

int s0, n, m;

int f[N][MAXN][MAXN];//状压数组，含义见题解

int cost[N];//雇第i个职员花多少钱
int teach[N];//第i个职员能交那些科目

int main()
{
	while(scanf("%d %d %d", &s0, &m, &n) != EOF)
	{
		if(s0 == 0) break;

		//数据不清空，爆零两行泪
		memset(f, 0x3f, sizeof(f));
		memset(teach , 0, sizeof(teach));

		//因为在职教师是必选的，那么可以提前把他们处理出来
		//把他们可以教的科目作为初始状态，花费的费用作为初始消耗
		int ori1 = 0 , ori2 = 0, mon = 0;
		//ori1与DP中的i等价，ori2与j等价，mon是代价
		for(int k = 1; k <= m; k++)
		{
			int c;
			scanf("%d", &c);
			mon += c;

			string s;
			getline(cin, s);
			//getline读整行

			int num = 0;
			for(int i = 0; i < (int )s.length() ; i++)
				if(s[i] >= '0' && s[i] <= '9')
					num *= 10, num += s[i] - '0';
				else if(num)
				{
					if(!(ori2 & (1 << (num - 1))))//这门课不到两个人教（两个人教与N个人教对于我们的状态来说，没有区别）
					{
						if(ori1 & ((1 << (num - 1))))//这门课有人一个教了， 那么把这一个人教的标志取消掉，换成两个人教的标志
							ori1 ^= ((1 << (num - 1))), ori2 |= ((1 << (num - 1)));
						else//否则，标记上这门课有一个人教
							ori1 |= ((1 << (num - 1)));
					}

					num = 0;
				}
			if(num)//最后一位的处理
			{
				if(!(ori2 & (1 << (num - 1))))
				{
					if(ori1 & ((1 << (num - 1))))
						ori1 ^= ((1 << (num - 1))), ori2 |= ((1 << (num - 1)));
					else
						ori1 |= ((1 << (num - 1)));
				}
			}
		}

		//初态设定
		f[0][ori1][ori2] = mon;

		for(int k = 1; k <= n; k++)
		{
			scanf("%d", &cost[k]);

			string s;
			getline(cin, s);

			int num = 0;
			for(int i = 0; i < (int )s.length() ; i++)
				if(s[i] >= '0' && s[i] <= '9')
					num *= 10, num += s[i] - '0';
				else if(num)
					teach[k] |= (1 << (num - 1)), num = 0;
			if(num)
				teach[k] |= (1 << (num - 1));
		}

		//我习惯的是刷表法...可能快一点
		for(int k = 0; k < n; k++)
			for(int i = 0; i < MAXN; i++)
				for(int j = 0; j < MAXN; j++)
					if(f[k][i][j] != 0x3f3f3f3f)
					{
						//不雇佣
						f[k + 1][i][j] = min(f[k + 1][i][j], f[k][i][j]);

						int to1 = i, to2 = j;
						//处理出雇佣之后的状态
						for(int c = 0; c < s0; c++)
						{
							//他不教这一科
							if(!(teach[k + 1] & (1 << c)))
								continue;

							//这科有两个人教了...
							if((j & (1 << c) ))
								continue;

							//这个处理有点类似之前读入的处理，不细讲
							if(i & (1 << c))
								to1 ^= (1 << c), to2 |= (1 << c);
							else
								to1 |= (1 << c);
						}
						//处理完毕，选择此人
						f[k + 1][to1][to2] = min(f[k + 1][to1][to2], f[k][i][j] + cost[k + 1]);
					}
		//末态：所有科目都有至少两个人教
		printf("%d\n", f[n][0][(1 << s0) - 1]);
	}

	return 0;
}
```
