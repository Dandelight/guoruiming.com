| 1h   | 聚会(AHOI 2008)                               | 提高  | https://www.luogu.com.cn/problem/P4281   |
| ---- | --------------------------------------------- | ----- | ---------------------------------------- |
| 1h   | Lightning Energy Report, Jakarta2010, UVa1674 | 提高  | https://www.luogu.com.cn/problem/UVA1674 |
| 1h   | Network(POJ3417)                              | 提高+ | http://poj.org/problem?id=3417           |
| 0.5h | USACO 2012 Dec G. Running Away From the Barn  | 提高+ | https://www.luogu.com.cn/problem/P3066   |
| 1h   | CFGYM102694C Sloth Naptime                    | 提高+ | https://codeforces.com/blog/entry/81527  |
| 1.5h | CF609E Minimum spanning tree for each edge    | 提高+ | https://www.luogu.com.cn/problem/CF609E  |

## 紧急集合

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 500010;
int n, m, s, head[N], num, t, dep[N], f[N][30];
struct node {
  int to, next;
} a[N * 2];
inline void add(int from, int to) {
  num++;
  a[num] = {to, head[from]};
  head[from] = num;
}
void dfs(int son, int fa) {
  dep[son] = dep[fa] + 1;
  f[son][0] = fa;
  for (int i = 1; i <= t; ++i) f[son][i] = f[f[son][i - 1]][i - 1];

  for (int i = head[son]; i; i = a[i].next) {
    int k = a[i].to;
    if (k != fa) dfs(k, son);
  }
}
int lca(int x, int y) {
  if (dep[x] > dep[y]) swap(x, y);
  for (int i = t; i >= 0; --i) {
    if (dep[f[y][i]] >= dep[x]) y = f[y][i];
  }
  if (x == y) return x;
  for (int i = t; i >= 0; --i) {
    if (f[x][i] != f[y][i]) {
      x = f[x][i];
      y = f[y][i];
    }
  }
  return f[x][0];
}
int main() {
  ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
  cin >> n >> m;
  t = log2(n);
  for (int i = 1; i <= n - 1; ++i) {
    int xa, xb;
    cin >> xa >> xb;
    add(xa, xb);
    add(xb, xa);
  }
  dfs(1, 0);
  int q, w,  // 深度最深的LCA对应的两个点
      e,     // 另外一个点
      zx;    // 深度最深的LCA
  for (int i = 1; i <= m; ++i) {
    int x, y, z;
    cin >> x >> y >> z;
    int l1 = lca(x, y), l2 = lca(x, z), l3 = lca(y, z);
    if (dep[l1] >= dep[l2] && dep[l1] >= dep[l3])
      q = x, w = y, e = z, zx = l1;
    else if (dep[l2] >= dep[l1] && dep[l2] >= dep[l3])
      q = x, w = z, e = y, zx = l2;
    else if (dep[l3] >= dep[l2] && dep[l3] >= dep[l2])
      q = z, w = y, e = x, zx = l3;

    int wdt = lca(q, e);  // 找到较浅的两个点的LCA
    int ans = dep[q] + dep[w] - 2 * dep[zx] + dep[e] + dep[zx] - 2 * dep[wdt];
    printf("%d %d\n", zx, ans);
  }
}
```

## Lightning Energy Report

```cpp
// 参考：<https://morris821028.github.io/2014/12/02/uva-1674/>
#include <bits/stdc++.h>
using namespace std;
const int N = 65536;
int visited[N];

// 并查集部分
int parent[N], setrank[N];

// 路径压缩 O(1)走起
int findp(int x) { return parent[x] == x ? x : (parent[x] = findp(parent[x])); }
int joint(int x, int y) {
  x = findp(x), y = findp(y);
  // 同一棵树上
  if (x == y) return 0;
  // x树比y树的秩大，将y树合并到x树上
  else if (setrank[x] > setrank[y])
    setrank[x] += setrank[y], parent[y] = x;
  // y树比x树的秩大，将x树合并到y树上
  else
    setrank[y] += setrank[x], parent[x] = y;
  return 1;
}

// LCA部分
vector<int> tree[N];              // 邻接图定义的树
vector<pair<int, int>> query[N];  // query 询问，<index, node>
int LCA[N];                       // 每次询问的两个点的LCA

/**
 * @brief tarjan算法是一种**离线算法**
 * 使用并查集记录某个节点的祖先节点
 * 在回溯过程中处理询问
 *
 * @param u 当前节点
 * @param p u节点的祖先节点
 */
void tarjan(int u, int p) {
  // 进行一次DFS遍历
  parent[u] = u;
  for (int i = 0; i < tree[u].size(); ++i) {
    int v = tree[u][i];
    if (v == p) continue;
    tarjan(v, u);
    // 记录当前节点的父节点
    parent[findp(v)] = u;
  }
  // 记录visited情况
  visited[u] = 1;
  // 回溯，以该节点为起点当遍历到某个节点时，认为根节点是其本身，
  // 对有关该节点的所有询问
  for (auto x : query[u]) {
    auto idx = x.first;
    auto node = x.second;
    // 如果已经访问过了，那么就是u节点的儿子
    if (visited[node]) {
      LCA[idx] = findp(node);
    }
  }
}

int childrenWeight[N],  // 子树（不包含根节点）权重之和
    rootWeight[N];      // 本节点权重之和

// 使用DFS统计权重
int dfs(int u, int p, int childrenWeight[]) {
  int sum = childrenWeight[u];
  for (auto v : tree[u]) {
    if (v == p) continue;
    sum += dfs(v, u, childrenWeight);
  }
  return childrenWeight[u] = sum;
}
int X[N], Y[N], K[N];
int main() {
  int n, m, x, y;
  int testcase;
  cin >> testcase;
  for (int cases = 1; cases <= testcase; ++cases) {
    cin >> n;
    for (int i = 0; i < n; ++i) tree[i].clear();
    for (int i = 1; i < n; ++i) {
      cin >> x >> y;
      tree[x].push_back(y);
      tree[y].push_back(x);
    }
    memset(childrenWeight, 0, sizeof(childrenWeight));
    memset(rootWeight, 0, sizeof rootWeight);
    memset(X, 0, sizeof(X));
    memset(Y, 0, sizeof(Y));
    memset(K, 0, sizeof(K));
    for (int i = 0; i < n; ++i) {
      visited[i] = 0, query[i].clear();
    }
    cin >> m;
    for (int i = 0; i < m; ++i) {
      cin >> X[i] >> Y[i] >> K[i];
      query[X[i]].emplace_back(i, Y[i]);
      query[Y[i]].emplace_back(i, X[i]);
    }

    // tarjan 法求LCA
    tarjan(0, -1);

    // 对每次雷击，都是给节点的子树增加K[i]，给节点减去K[i]
    for (int i = 0; i < m; ++i) {
      rootWeight[LCA[i]] += K[i];
      childrenWeight[X[i]] += K[i];
      childrenWeight[Y[i]] += K[i];
      childrenWeight[LCA[i]] -= 2 * K[i];
    }

    // dfs 统计电量
    dfs(0, -1, childrenWeight);

    printf("Case #%d:\n", cases);
    for (int i = 0; i < n; ++i)
      printf("%d\n", childrenWeight[i] + rootWeight[i]);
  }
}
```

## Network

```cpp
#include <cmath>
#include <cstdio>
#include <queue>

using namespace std;
const int N = 1e5 + 5, M = N * 2;
struct E {
  int v, next;
} e[M];
int h[N];
// 以上 链表模板

int len, ans, lg,
    f[N][20],  // 倍增法预处理的cost数组
    dep[N],    // 用于bfs，兼具visited数组的功能
    d[N];
// 以上 倍增法求LCA子树的总权值
int n, m;
void add(int u, int v) {
  e[++len].v = v;
  e[len].next = h[u];
  h[u] = len;
}
void bfs(int start) {
  queue<int> q;
  dep[start] = 1;
  q.push(start);
  while (!q.empty()) {
    int u = q.front();
    q.pop();
    for (int j = h[u]; j; j = e[j].next) {
      int v = e[j].v;
      if (dep[v]) continue;
      dep[v] = dep[u] + 1;
      q.push(v);
      f[v][0] = u;
      for (int k = 1; k <= lg; ++k) f[v][k] = f[f[v][k - 1]][k - 1];
    }
  }
}
// 求 x 与 y 的 lca
int lca(int x, int y) {
  // 令 x 比 y 深
  if (dep[y] > dep[x]) swap(x, y);
  // 令 y 和 x 在同一个深度
  for (int k = lg; k >= 0; --k) {
    if (dep[f[x][k]] >= dep[y]) x = f[x][k];
  }
  // 如果这个时候 y = x，那么 x，y 就都是它们自己的祖先。
  if (x == y) return x;
  // 不然的话，找到第一个不是它们祖先的两个点。
  for (int k = lg; k >= 0; --k) {
    if (f[x][k] != f[y][k]) x = f[x][k], y = f[y][k];
  }
  // 返回结果
  return f[x][0];
}

void dfs(int u, int fa) {
  for (int j = h[u]; j; j = e[j].next) {
    int v = e[j].v;
    if (v == fa) continue;
    dfs(v, u);
    // 这条边不在任何环中，去掉它就能分成两部分，这时候可以去掉m条额外边中任意一条
    if (d[v] == 0)
      ans += m;
    // 这条边e1在有且仅有一条额外边e2的环中，去掉e1和e2
    else if (d[v] == 1)
      ans += 1;
    // 没有其他可以增加d数组的方式了
    // 祖先的环数等于子节点环数之和
    d[u] += d[v];
  }
}

int main() {
  scanf("%d%d", &n, &m);
  lg = int(log(n) / log(2)) + 1;
  for (int i = 1; i < n; ++i) {
    int u, v;
    scanf("%d%d", &u, &v);
    add(u, v), add(v, u);
  }
  bfs(1);
  for (int i = 1; i <= m; ++i) {
    int u, v;
    scanf("%d%d", &u, &v);
    int LCA = lca(u, v);
    // 额外加一条路径构成一个环，让环上树边都+1
    // 显然全部+1不现实，采用树上差分方式
    d[LCA] -= 2, d[u] += 1, d[v] += 1;
  }
  dfs(1, 0);
  printf("%d\n", ans);
}
```

## Running away from the barn

```cpp
/**
 * @file main.cpp
 * @author Ruiming Guo (guoruiming@stu.scu.edu.cn)
 * @brief 给定一颗 n 个点的有根树，边有边权，节点从 1 至 n 编号，1
 * 号节点是这棵树的根。 再给出一个参数 t，对于树上的每个节点 u，请求出 u
 * 的子树中有多少节点满足该节点到 u 的距离不大于 t。
 * @version 0.1
 * @date 2022-04-29
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 200005;
int n;
ll m;
int cnt, head[N];
struct node {
  int to, next;
} e[N];
// 以上 链式前向星
int idx;
int poi[N];    // dfs欧拉序列
int f[N][23];  // f[i][j]: 第i个结点向上跳2^j个结点后到达的结点
ll dis[N];     // 距离
int dlt[N];    // 答案
void add(int u, int v) {
  e[++cnt] = {v, head[u]};
  head[u] = cnt;
}
int find(int x) {
  int now = x;
  for (int j = 20; j >= 0; --j)
    if (dis[x] - dis[f[now][j]] <= m) now = f[now][j];
  return f[now][0];
}
void dfs(int x) {
  idx++;         // 时间戳
  poi[idx] = x;  // 欧拉序列
  for (int i = head[x]; i; i = e[i].next) dfs(e[i].to);
}
int main() {
  scanf("%d%lld", &n, &m);
  dlt[1] = 1;
  // 对每个节点
  for (int i = 2; i <= n; ++i) {
    int x;
    ll w;
    scanf("%d%lld", &x, &w);
    add(x, i);
    // 边权化为点权
    f[i][0] = x;
    dis[i] = dis[x] + w;
    // 为了代码清楚把初始化fa数组放到dfs前边了
    // for (int j = 1; j <= 20; ++j) f[i][j] = f[f[i][j - 1]][j - 1];
    // 树上差分
    dlt[i]++;
    dlt[find(i)]--;
  }

  // 这行跟数据没关系，就是在建立fa倍增数组
  for (int i = 2; i <= n; ++i)
    for (int j = 1; j <= 20; ++j) f[i][j] = f[f[i][j - 1]][j - 1];

  dfs(1);

  // 将统计子树中合法节点的个数(top-down)，改为从每个节点，将合法的祖先节点加上1(bottom-up)
  for (int i = n; i; i--) dlt[f[poi[i]][0]] += dlt[poi[i]];
  for (int i = 1; i <= n; ++i) printf("%d\n", dlt[i]);
  return 0;
}
```

## Sloth Naptime

附上官方`java`答案

```java
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.StringTokenizer;
public class TreeBasicsSloth {

	public static void main(String[] args) {
		FastScanner fs=new FastScanner();
		int n=fs.nextInt();
		Node[] nodes=new Node[n];
		for (int i=0; i<n; i++) nodes[i]=new Node(i+1);
		for (int i=1; i<n; i++) {
			int a=fs.nextInt()-1, b=fs.nextInt()-1;
			nodes[a].adj.add(nodes[b]);
			nodes[b].adj.add(nodes[a]);
		}
		nodes[0].dfs0(null, 0);
		for (int e=1; e<20; e++)
			for (Node nn:nodes)
				if (nn.lift[e-1]!=null)
					nn.lift[e]=nn.lift[e-1].lift[e-1];

		int q=fs.nextInt();
		PrintWriter out=new PrintWriter(System.out);
		for (int qq=0; qq<q; qq++) {
			Node a=nodes[fs.nextInt()-1], b=nodes[fs.nextInt()-1];
			int c=fs.nextInt();
			Node lca=a.lca(b, 19);
			int totalDist=a.depth+b.depth-lca.depth*2;
			if (totalDist<=c) {
				out.println(b);
				continue;
			}

			int aDist=a.depth-lca.depth;
			if (c<=aDist) {
				out.println(a.goUp(c).id);
			}
			else {
				int bUp=totalDist-c;
				out.println(b.goUp(bUp));
			}
		}
		out.close();
	}

	static class Node {
		Node[] lift=new Node[20];
		int depth, id;
		ArrayList<Node> adj=new ArrayList<>();

		public Node(int id) {
			this.id=id;
		}

		public void dfs0(Node par, int depth) {
			if (par!=null) adj.remove(par);
			this.depth=depth;
			lift[0]=par;
			for (int i=0; i<adj.size(); i++) {
				if (adj.get(i)==par) continue;
				adj.get(i).dfs0(this, depth+1);
			}
		}

		public Node goUp(int nSteps) {
			if (nSteps==0) return this;
			return lift[Integer.numberOfTrailingZeros(nSteps)].goUp(nSteps-Integer.lowestOneBit(nSteps));
		}

		public Node lca(Node o, int nJumps) {
			if (this==o) return this;
			if (depth!=o.depth) {
				if (depth>o.depth) return goUp(depth-o.depth).lca(o, 19);
				return lca(o.goUp(o.depth-depth), 19);
			}
			if (lift[0]==o.lift[0]) return lift[0];
			while (lift[nJumps]==o.lift[nJumps]) nJumps--;
			return lift[nJumps].lca(o.lift[nJumps], nJumps);
		}

		public String toString() {
			return id+"";
		}

	}

	static class FastScanner {
		BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
		StringTokenizer st=new StringTokenizer("");
		public String next() {
			while (!st.hasMoreElements())
				try {
					st=new StringTokenizer(br.readLine());
				} catch (IOException e) {
					e.printStackTrace();
				}
			return st.nextToken();
		}

		int nextInt() {
			return Integer.parseInt(next());
		}
	}
}
```

## Minimum spanning tree for each edge

```cpp
/**
 * @file main.cpp
 * @author Ruiming Guo (guoruiming@stu.scu.edu.cn)
 * @brief 给定一棵无重边无自环的树，给定树中一条边，求包含该边的最小的生成树
 *
 * 可以如此考虑：
 *
 * 设给定的边为 (x, y)，先使用 Kruskal 算法找到 MST ，再去掉连接节点 x 和 y
 * 的路径上最重的一条边， 最后把 (x, y) 那条边加上。
 *
 * 找到最重的边可以先找树根到LCA(x, y)的最重的边，再找LCA(x,y)到 x 和 y
 * 中最重的边，取三者最大值。
 *
 * 时间复杂度：O(M*log(N))
 *
 * @version 0.1
 * @date 2022-04-29
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <bits/stdc++.h>
using namespace std;
const int N = 210000;
const int L = 17;
int n, m;
vector<pair<int, int>> g[N];
vector<tuple<int, int, int>> edges, edges2, mst;
int w[N];
long long cost;
int timer;
int up[N][L + 1];
int bedge[N][L + 1];
int tin[N], tout[N], dep[N];

void dfs(int v, int p = 1, int pcost = 0) {
  tin[v] = timer++;
  up[v][0] = p;
  bedge[v][0] = pcost;
  for (int i = 1; i <= L; ++i) {
    up[v][i] = up[up[v][i - 1]][i - 1];
    bedge[v][i] = max(bedge[v][i - 1], bedge[up[v][i - 1]][i - 1]);
  }
  for (auto [to, cost] : g[v]) {
    if (to == p) continue;
    dep[to] = dep[v] + 1;
    dfs(to, v, cost);
  }
  tout[v] = timer++;
}

int get(int x) {
  if (x == w[x]) return x;
  return w[x] = get(w[x]);
}

bool upper(int a, int b) {
  return (tin[a] <= tin[b] && tout[a] >= tout[b]);  // 去掉等号是不是也可以
}

int lca(int a, int b) {
  if (upper(a, b)) return a;
  if (upper(b, a)) return b;
  for (int i = L; i >= 0; --i) {
    if (!upper(up[a][i], b)) a = up[a][i];
  }
  return up[a][0];
}

void merge(int a, int b) {
  if (rand() % 2) swap(a, b);
  a = get(a), b = get(b);
  w[a] = b;
}

int get_best(int v, int span) {
  int ret = 0;
  for (int i = L; i >= 0; --i) {
    if (span & (1 << i)) {
      ret = max(ret, bedge[v][i]);
      v = up[v][i];
    }
  }
  return ret;
}

int main() {
  cin >> n >> m;
  for (int i = 1; i <= m; ++i) {
    int a, b, c;
    cin >> a >> b >> c;
    edges.push_back({c, a, b});
    edges2.push_back({c, a, b});
  }
  sort(edges.begin(), edges.end());
  for (int i = 1; i <= n; ++i) w[i] = i;
  for (auto [c, a, b] : edges) {
    int ta = get(a), tb = get(b);
    if (ta == tb) continue;
    merge(ta, tb);
    mst.push_back({c, a, b});
    cost += c;
  }
  for (auto [cost, v1, v2] : mst) {
    g[v1].push_back({v2, cost});
    g[v2].push_back({v1, cost});
  }
  dfs(1);
  for (auto [c, v1, v2] : edges2) {
    int l = lca(v1, v2);
    int bst = 0;
    bst = max(
        {bst, get_best(v1, dep[v1] - dep[l]), get_best(v2, dep[v2] - dep[l])});
    cout << cost + c - bst << '\n';
  }
}
```
