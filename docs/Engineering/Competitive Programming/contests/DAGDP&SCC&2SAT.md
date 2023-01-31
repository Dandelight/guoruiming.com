|       |                                       |       |                                                           |
| ----- | ------------------------------------- | ----- | --------------------------------------------------------- |
| DAGDP | UVa357 Let Me Count The Ways          | 提高- | https://www.luogu.com.cn/problem/UVA357                   |
| DAGDP | The Tower of Babylon, POJ2241         | 提高+ | https://www.luogu.com.cn/problem/UVA437                   |
| DAGDP | A Spy in the Metro, WF2003, UVa1025   | 提高+ | https://www.luogu.com.cn/problem/UVA1025                  |
| DAGDP | USACO 2020 Feb G. Timeline            | 提高- | https://www.luogu.com.cn/problem/P6145                    |
| DAGDP | CF919D Substring                      | 提高- | https://codeforces.com/contest/919/problem/D              |
| DAGDP | CF510C. Fox And Names                 | 提高  | https://codeforces.com/problemset/problem/510/C           |
| SCC   | POJ2186 受欢迎的牛(USACO 2003 Fall)   | 提高  | https://www.luogu.com.cn/problem/P2341                    |
| SCC   | 最大团 (The Largest Clique, UVa11324) | 提高  | https://www.luogu.com.cn/problem/UVA11324                 |
| SCC   | 最大半连通子图 ZJOI2007               | 提高+ | https://www.luogu.com.cn/problem/P2272                    |
| DAGDP | Kattis Quantum Superposition          | 提高+ | https://open.kattis.com/problems/quantumsuperposition     |
| DAGDP | USACO 2018 Open G. Milking Order      | 提高+ | http://www.usaco.org/index.php?page=viewproblem2&cpid=838 |
| SCC   | Proving Equivalences, HDU2767         | 提高+ | http://acm.hdu.edu.cn/showproblem.php?pid=2767            |
| SCC   | [HAOI2010]软件安装                    | NOI-  | https://www.luogu.com.cn/problem/P2515                    |
| 2-SAT | 宇航员分组 (Astronauts, UVa1391)      | NOI-  | https://www.luogu.com.cn/problem/UVA1391                  |
| 2-SAT | 飞机调度 (Now or later, UVa1146)      | NOI-  | https://www.luogu.com.cn/problem/UVA1146                  |
| SCC   | USACO 2015 JanG Grass Cownoisseur     | NOI-  | http://www.usaco.org/index.php?page=viewproblem2&cpid=516 |
| SCC   | CF1239D Catowice City                 | 提高+ | https://codeforces.com/contest/1239/problem/D             |
| SCC   | POI2012 – Festival                    | NOI-  | https://www.luogu.com.cn/problem/P3530                    |

## Let Me Count The Ways

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 30010;
typedef long long ll;
ll arr[N];
const int coin[] = {1, 5, 10, 25, 50};
int main() {
  int x;
  arr[0] = 1;
  for (auto x : coin) {
    for (int i = 0; i <= 30000; ++i) {
      arr[i + x] += arr[i];
    }
  }
  while (cin >> x) {
    ll ans = arr[x];
    if (ans != 1) {
      printf("There are %lld ways to produce %d cents change.\n", ans, x);
    } else {
      printf("There is only 1 way to produce %d cents change.\n", x);
    }
  }
}
```

## The Tower of Babylon

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 100;
int n, m, id;
struct cube {
  int x, y, h;
  bool operator>(const cube& b) const {
    return (x > b.x && y > b.y) || (x > b.y && y > b.x);
  }
} c[N];
int dp[N];
int get(int id) {
  int& ans = dp[id];
  if (ans > 0) return ans;
  ans = c[id].h;
  for (int i = 1; i <= m; ++i) {
    if (c[id] > c[i]) {
      ans = max(ans, get(i) + c[id].h);
    }
  }
  return ans;
}
int main() {
  int tc = 0;
  while (cin >> n && n) {
    memset(dp, 0, sizeof(dp));
    m = n * 3;
    for (int i = 1; i <= m; i += 3) {
      int x, y, h;
      cin >> x >> y >> h;
      c[i] = {x, y, h};
      c[i + 1] = {h, x, y};
      c[i + 2] = {y, h, x};
    }
    int ans = 0;
    for (int i = 1; i <= m; i++) {
      ans = max(get(i), ans);
    }
    printf("Case %d: maximum height = %d\n", ++tc, ans);
  }
}
```

## A Spy in the Metro

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int ti[55];  // 每辆火车行驶一站的时间
// has_train[t][i][0]：在t时刻i车站是否有向右的火车
bool has_train[2005][55][2];
// dp[i][j]：在i时刻，你在车站j最少还需要等待多长时间
int dp[2055][55];
int main() {
  int tc = 0;
  int n;
  while (cin >> n && n) {
    int t;
    cin >> tac;
    memset(has_train, 0, sizeof(has_train));
    for (int i = 1; i <= n - 1; ++i) {
      cin >> ti[i];
    }
    int m1;
    cin >> m1;
    for (int i = 0; i < m1; ++i) {
      // rightward
      int d;
      cin >> d;
      for (int j = 1; j <= n; ++j) {
        if (d > t) break;
        has_train[d][j][0] = true;
        d += ti[j];
      }
    }
    int m2;
    cin >> m2;
    for (int i = 0; i < m2; ++i) {
      // leftward
      int e;
      cin >> e;
      for (int j = n; j >= 1; --j) {
        if (e > t) break;
        has_train[e][j][1] = true;
        e += ti[j - 1];  // leftward
      }
    }
    for (int j = 0; j < n; ++j) dp[t][j] = 5000;
    dp[t][n] = 0;
    for (int i = t - 1; i >= 0; --i) {
      for (int j = 1; j <= n; ++j) {
        dp[i][j] = dp[i + 1][j] + 1;
        // rightward
        if (j < n && i + ti[j] <= t && has_train[i][j][0]) {
          dp[i][j] = min(dp[i][j], dp[i + ti[j]][j + 1]);
        }
        // leftward
        if (j > 1 && i + ti[j - 1] <= t && has_train[i][j][1]) {
          dp[i][j] = min(dp[i][j], dp[i + ti[j - 1]][j - 1]);
        }
      }
    }
    printf("Case Number %d: ", ++tc);
    if (dp[0][1] >= 5000)
      printf("impossible\n");
    else
      printf("%d\n", dp[0][1]);
  }
}
```

## Timeline

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef pair<int, int> pi;
const int N = 100000 + 5;
vector<pi> adj[N];
int s[N], t[N];
int main() {
  int n, m, c;
  queue<int> q;
  cin >> n >> m >> c;
  for (int i = 1; i <= n; ++i) cin >> s[i];
  for (int i = 1; i <= c; ++i) {
    int u, v, w;
    cin >> u >> v >> w;
    adj[u].emplace_back(v, w);
    t[v]++;
  }
  // 拓扑排序
  for (int i = 1; i <= n; ++i)
    if (!t[i]) q.push(i);
  while (!q.empty()) {
    int u = q.front();
    q.pop();
    for (auto [v, w] : adj[u]) {
      s[v] = max(s[v], s[u] + w);
      t[v]--;
      if (!t[v]) q.push(v);
    }
  }
  for (int i = 1; i <= n; ++i) cout << s[i] << '\n';
}
```

## Substring

```cpp
// 拓扑排序可以排非连通图？！

#include <bits/stdc++.h>
using namespace std;
const int N = 300'000 + 10;
// dp[i][alpha] 表示在字符串前i个字符组成的图中
// alpha这个字符的最大数量
int dp[N][26];
char s[N];
int indeg[N];
vector<int> adj[N];
int m, n;
int main() {
  cin >> n >> m >> s + 1;
  for (int i = 0; i < m; ++i) {
    int u, v;
    cin >> u >> v;
    indeg[v]++;
    adj[u].push_back(v);
  }
  queue<int> q;
  int cnt = 0;
  for (int i = 1; i <= n; ++i) {
    if (indeg[i] == 0) {
      q.push(i);
      dp[i][s[i] - 'a'] = 1;
    }
  }
  while (q.size()) {
    int u = q.front();
    q.pop();
    cnt++;
    for (auto v : adj[u]) {
      for (int j = 0; j < 26; ++j) {
        dp[v][j] = max(dp[v][j], dp[u][j] + (s[v] - 'a' == j));
      }

      indeg[v]--;
      if (indeg[v] == 0) q.push(v);
    }
  }

  int ans = -1;
  // 因为这道题特殊所以找答案需要找遍dp数组
  if (cnt == n) {
    for (int i = 1; i <= n; ++i) {
      for (int j = 0; j < 26; ++j) {
        ans = max(ans, dp[i][j]);
      }
    }
  }

  cout << ans << endl;
}
```

## Fox and Names

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 110;
const int A = 30;
int n;
vector<int> adj[A];
int indeg[A];
string name[N];
int enc(int a) {
  if (a == ' ')
    return 0;
  else
    return a - 'a' + 1;
}
int main() {
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> name[i];
    name[i] += ' ';
  }
  for (int i = 0; i < n - 1; ++i) {
    int pos = 0;
    while (name[i][pos] == name[i + 1][pos]) pos++;
    int u = enc(name[i][pos]), v = enc(name[i + 1][pos]);
    if (v == 0) {  // xy > x
      cout << "Impossible\n";
      return 0;
    }
    adj[u].push_back(v);
    indeg[v]++;
  }
  queue<int> q;
  int cnt = 0;
  for (int i = 0; i <= 26; ++i) {
    if (!indeg[i]) q.push(i);
  }
  string ans;
  while (q.size()) {
    int u = q.front();
    q.pop();
    if (u) {
      ans += char(u + 'a' - 1);
      cnt++;
    }
    // cout << ans << endl;
    for (int v : adj[u]) {
      indeg[v]--;
      if (!indeg[v]) q.push(v);
    }
  }
  if (cnt == 26)
    cout << ans << endl;
  else
    cout << "Impossible\n";
}
```

## 受欢迎的牛 G

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 1e4 + 10, M = 5e4 + 10;
vector<int> adj[N];
int dfn[N], low[N], deg[N], sz[N];
int scc[N], sccidx;  // 节点i所在SCC的编号
bool instack[N];
int n, m;
int dfncnt;
stack<int> s;
void tarjan(int u) {
  dfn[u] = low[u] = ++dfncnt;
  s.push(u);
  instack[u] = true;
  for (auto v : adj[u]) {
    if (!dfn[v]) {  // v未被搜索过
      tarjan(v);
      low[u] = min(low[u], low[v]);
    } else if (instack[v])  // v在栈中
      low[u] = min(low[u], dfn[v]);
  }
  int k = -1;  // 不必担心，k在使用前一定是赋值了的
  if (low[u] == dfn[u]) {
    ++sccidx;
    do {
      k = s.top();
      s.pop();
      instack[k] = false;
      scc[k] = sccidx;
      sz[sccidx]++;
    } while (u != k);
  }
}
int main() {
  cin >> n >> m;
  for (int i = 0; i < m; ++i) {
    int u, v;
    cin >> u >> v;
    adj[u].push_back(v);
  }
  for (int i = 1; i <= n; ++i) {
    if (!dfn[i]) tarjan(i);
  }
  for (int u = 1; u <= n; ++u) {
    for (auto v : adj[u]) {
      if (scc[u] != scc[v]) {
        deg[scc[u]]++;  // 遍历点并记录出度
      }
    }
  }

  bool id = 0;
  for (int i = 1; i <= sccidx; ++i) {
    if (!deg[i]) {
      if (id) {
        // 两次出现出度为0的点，直接输出0
        puts("0");
        return 0;
      }
      id = i;
    }
  }

  cout << sz[id] << endl;
}
```

## The Largest Clique

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 10010;
vector<int> adj[N], sccadj[N];
int dfn[N], low[N], deg[N], sz[N], scc[N], memo[N], dfncnt, scccnt, ans;
bool instk[N];
int n, m;
stack<int> s;

void tarjan(int u) {
  dfn[u] = low[u] = ++dfncnt;
  s.push(u);
  instk[u] = true;
  for (auto v : adj[u]) {
    if (!dfn[v]) {
      tarjan(v);
      low[u] = min(low[u], low[v]);
    } else if (instk[v]) {
      low[u] = min(low[u], dfn[v]);
    }
  }
  if (dfn[u] == low[u]) {
    scccnt++;
    int k = -1;
    do {
      k = s.top();
      s.pop();
      sz[scccnt]++;
      instk[k] = false;
      scc[k] = scccnt;
    } while (k != u);
  }
}

int dfs(int u) {
  if (memo[u]) return memo[u];
  for (const int v : sccadj[u]) memo[u] = max(memo[u], dfs(v));
  return memo[u] = memo[u] + sz[u];
}

int main() {
  int tc;
  cin >> tc;
  while (tc--) {
    cin >> n >> m;

    for (int *arr : {dfn, low, deg, sz, scc, memo}) {
      memset(arr, 0, sizeof(dfn));
    }
    memset(instk, 0, sizeof instk);
    while (s.size()) s.pop();
    dfncnt = scccnt = ans = 0;
    for (auto &v : {adj, sccadj})
      for (int i = 1; i <= n; ++i) v[i].clear();

    for (int i = 1; i <= m; ++i) {
      int u, v;
      cin >> u >> v;
      adj[u].push_back(v);
    }
    for (int i = 1; i <= n; ++i)
      if (!dfn[i]) tarjan(i);

    for (int u = 1; u <= n; ++u) {
      for (int v : adj[u]) {
        if (scc[u] != scc[v]) {
          sccadj[scc[u]].push_back(scc[v]);
        }
      }
    }
    for (int i = 1; i <= scccnt; ++i) {
      ans = max(ans, dfs(i));
    }
    cout << ans << endl;
  }
}

```

## 最大半连通子图

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 100010;
int dfn[N], low[N], deg[N], sz[N], scc[N], dfncnt, scccnt;
bool instk[N];
vector<int> adj[N], sccadj[N];
stack<int> s;
int f[N], g[N], used[N];
int n, m, x;
void tarjan(int u) {
  dfn[u] = low[u] = ++dfncnt;
  s.push(u);
  instk[u] = true;
  for (auto v : adj[u]) {
    if (!dfn[v]) {
      tarjan(v);
      low[u] = min(low[u], low[v]);
    } else if (instk[v]) {
      low[u] = min(low[u], dfn[v]);
    }
  }
  if (dfn[u] == low[u]) {
    scccnt++;
    int k = -1;
    do {
      k = s.top();
      s.pop();
      sz[scccnt]++;
      instk[k] = false;
      scc[k] = scccnt;
    } while (k != u);
  }
}
int main() {
  cin >> n >> m >> x;
  for (int i = 0; i < m; ++i) {
    int u, v;
    cin >> u >> v;
    adj[u].push_back(v);
  }
  for (int i = 1; i <= n; ++i)
    if (!dfn[i]) tarjan(i);

  for (int u = 1; u <= n; ++u) {
    f[u] = sz[u];
    g[u] = 1;
    for (auto v : adj[u]) {
      if (scc[u] == scc[v]) continue;
      sccadj[scc[u]].push_back(scc[v]);
    }
  }
  for (int u = scccnt; u >= 1; --u) {
    for (int v : sccadj[u]) {
      if (used[v] == u) continue;
      used[v] = u;
      if (f[v] < f[u] + sz[v]) {
        f[v] = f[u] + sz[v];
        g[v] = g[u];
      } else if (f[v] == f[u] + sz[v]) {
        g[v] += g[u];
        g[v] %= x;
      }
    }
  }
  int ans = 0, tmp = 0;
  for (int i = 1l; i <= scccnt; ++i) {
    if (f[i] > ans) {
      ans = f[i];
      tmp = g[i];
    } else if (f[i] == ans) {
      tmp += g[i];
      tmp %= x;
    }
  }
  cout << ans << endl << tmp << endl;
}
```

## Quantum Superposition

```cpp
#include <bits/stdc++.h>

using namespace std;
const int N = 1000 + 10;
typedef long long ll;

vector<int> adj1[N], adj2[N];
vector<int> dis1, dis2;
bool nm[N * 2];
bool vis[N][N];
int n1, n2, m1, m2;

void dfs(vector<int> &dis, int n, vector<int> (&adj)[N], int u, int dep) {
  if (u == n) dis.push_back(dep);
  for (int v : adj[u]) {
    if (!vis[v][dep + 1]) {
      vis[v][dep + 1] = true;
      dfs(dis, n, adj, v, dep + 1);
    }
  }
}

int main() {
  cin >> n1 >> n2 >> m1 >> m2;
  for (int i = 1; i <= m1; i++) {
    int x, y;
    cin >> x >> y;
    adj1[x].push_back(y);
  }
  for (int i = 1; i <= m2; i++) {
    int x, y;
    cin >> x >> y;
    adj2[x].push_back(y);
  }
  dfs(dis1, n1, adj1, 1, 0);
  memset(vis, 0, sizeof(vis));
  dfs(dis2, n2, adj2, 1, 0);
  for (int x : dis1) {
    for (int y : dis2) {
      nm[x + y] = true;
    }
  }
  int k, q;
  cin >> k;
  while (k--) {
    cin >> q;
    cout << (nm[q] ? "YES\n" : "NO\n");
  }
  return 0;
}
```

## Milking Order G

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 1e5 + 5;
int n, m;
vector<int> adj[N];
vector<int> vec[N];
int indeg[N];
// 考虑前x条信息时的图
void build(int x) {
  for (int i = 0; i <= n; ++i) adj[i].clear();
  memset(indeg, 0, sizeof indeg);
  for (int i = 1; i <= x; ++i) {
    for (int j = 0; j < vec[i].size() - 1; ++j) {
      int u = vec[i][j], v = vec[i][j + 1];
      adj[u].push_back(v);
      indeg[v]++;
    }
  }
}

bool check(int x) {
  build(x);
  queue<int> q;
  int cnt = 0;
  for (int i = 1; i <= n; ++i)
    if (!indeg[i]) q.push(i);
  while (q.size()) {
    int u = q.front();
    q.pop();
    cnt++;
    for (int v : adj[u]) {
      indeg[v]--;
      if (!indeg[v]) q.push(v);
    }
  }
  return cnt == n;
}

void get_ans(int x) {
  build(x);
  priority_queue<int, vector<int>, greater<int>> q;
  for (int i = 1; i <= n; ++i)
    if (!indeg[i]) q.push(i);
  while (q.size()) {
    int u = q.top();
    q.pop();
    cout << u << ' ';
    for (int v : adj[u]) {
      indeg[v]--;
      if (!indeg[v]) q.push(v);
    }
  }
  cout << endl;
}

int main() {
  cin >> n >> m;
  for (int i = 1; i <= m; ++i) {
    int si;
    cin >> si;
    for (int j = 1; j <= si; ++j) {
      int x;
      cin >> x;
      vec[i].push_back(x);
    }
  }
  int l = 1, r = m, ans;
  while (l <= r) {
    int mid = (l + r) / 2;
    if (check(mid))
      ans = mid, l = mid + 1;
    else
      r = mid - 1;
  }
  get_ans(ans);
}
```

## Proving Equivalences

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 2e4 + 10;
int dfn[N], low[N], deg[N], sz[N], scc[N], outdeg[N], indeg[N], dfncnt, scccnt;
bool instk[N];
vector<int> adj[N];
stack<int> s;
int n, m;
void tarjan(int u) {
  dfn[u] = low[u] = ++dfncnt;
  s.push(u);
  instk[u] = true;
  for (auto v : adj[u]) {
    if (!dfn[v]) {
      tarjan(v);
      low[u] = min(low[u], low[v]);
    } else if (instk[v]) {
      low[u] = min(low[u], dfn[v]);
    }
  }
  if (dfn[u] == low[u]) {
    scccnt++;
    int k = -1;
    do {
      k = s.top();
      s.pop();
      sz[scccnt]++;
      instk[k] = false;
      scc[k] = scccnt;
    } while (k != u);
  }
}
int main() {
  int tc;
  cin >> tc;
  while (tc--) {
    cin >> n >> m;
    for (int *arr : {dfn, low, deg, sz, scc, outdeg, indeg})
      memset(arr, 0, sizeof dfn);
    memset(instk, 0, sizeof instk);
    while (s.size()) s.pop();
    dfncnt = scccnt = 0;
    for (int i = 1; i <= n; ++i) adj[i].clear();
    for (int i = 1; i <= m; ++i) {
      int u, v;
      cin >> u >> v;
      adj[u].push_back(v);
    }
    for (int i = 1; i <= n; ++i)
      if (!dfn[i]) tarjan(i);

    for (int u = 1; u <= n; ++u)
      for (int v : adj[u])
        if (scc[u] != scc[v]) ++outdeg[scc[u]], ++indeg[scc[v]];

    if (scccnt == 1) {
      cout << 0 << endl;
    } else {
      int cntin = 0, cntout = 0;
      for (int i = 1; i <= scccnt; ++i) {
        if (!indeg[i]) ++cntin;
        if (!outdeg[i]) ++cntout;
      }
      cout << max(cntin, cntout) << endl;
    }
  }
}
```

## 软件安装

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 510;
int dfn[N], low[N], deg[N], sz[N], scc[N], outdeg[N], indeg[N], dfncnt, scccnt;
bool instk[N];
vector<int> adj[N];
int w[N], a[N], d[N];
int W[N], V[N], dp[N][N];
stack<int> s;
int n, m;
void tarjan(int u) {
  dfn[u] = low[u] = ++dfncnt;
  s.push(u);
  instk[u] = true;
  for (auto v : adj[u]) {
    if (!dfn[v]) {
      tarjan(v);
      low[u] = min(low[u], low[v]);
    } else if (instk[v]) {
      low[u] = min(low[u], dfn[v]);
    }
  }
  if (dfn[u] == low[u]) {
    scccnt++;
    int k = -1;
    do {
      k = s.top();
      s.pop();
      sz[scccnt]++;
      instk[k] = false;
      scc[k] = scccnt;
      W[scccnt] += w[k];
      V[scccnt] += a[k];
    } while (k != u);
  }
}
void solve(int u) {
  for (int i = W[u]; i <= m; ++i) {
    dp[u][i] = V[u];
  }
  for (auto v : adj[u]) {
    solve(v);
    int k = m - W[u];
    for (int i = k; i >= 0; --i) {
      for (int j = 0; j <= i; ++j) {
        dp[u][i + W[u]] = max(dp[u][i + W[u]], dp[v][j] + dp[u][i + W[u] - j]);
      }
    }
  }
}
int main() {
  cin >> n >> m;
  for (int i = 1; i <= n; ++i) cin >> w[i];
  for (int i = 1; i <= n; ++i) cin >> a[i];
  for (int i = 1; i <= n; ++i) {
    cin >> d[i];
    if (d[i]) adj[d[i]].push_back(i);
  }
  for (int i = 1; i <= n; ++i)
    if (!dfn[i]) tarjan(i);
  for (int i = 0; i <= n; ++i) adj[i].clear();
  for (int i = 1; i <= n; ++i) {
    if (scc[d[i]] != scc[i]) {
      adj[scc[d[i]]].push_back(scc[i]);
      indeg[scc[i]]++;
    }
  }
  for (int i = 1; i <= scccnt; ++i) {
    if (!indeg[i]) adj[0].push_back(i);
  }

  solve(0);
  cout << dp[0][m] << endl;
}
```

## Astronauts

```cpp
#include <bits/stdc++.h>
using namespace std;

const int maxn = 200100, maxm = 400100;

int n, m, head[maxn], len, age[maxn];
bool vis[maxn];
queue<int> q;
struct edge {
    int to, next;
} edges[maxm];

void add_edge(int u, int v) {
    edges[++len].to = v;
    edges[len].next = head[u];
    head[u] = len;
}

void add_sub(int u, int a, int v, int b) {
    add_edge(u + n * (a ^ 1), v + n * b);
    add_edge(v + n * (b ^ 1), u + n * a);
}

bool dfs(int u) {
    if (vis[u + n * (u <= n ? 1 : -1)]) {
        return 0;
    }
    if (vis[u]) {
        return 1;
    }
    vis[u] = 1;
    q.push(u);
    for (int i = head[u]; i; i = edges[i].next) {
        int v = edges[i].to;
        if (!dfs(v)) {
            return 0;
        }
    }
    return 1;
}

bool twoSAT() {
    for (int i = 1; i <= n; ++i) {
        if (!vis[i] && !vis[i + n]) {
            while (q.size()) {
                q.pop();
            }
            if (!dfs(i)) {
                while (q.size()) {
                    vis[q.front()] = 0;
                    q.pop();
                }
                if (!dfs(i + n)) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

int main() {
    while (scanf("%d%d", &n, &m) == 2 && n) {
        len = 0;
        memset(vis, 0, sizeof(vis));
        memset(head, 0, sizeof(head));
        int sum = 0;
        for (int i = 1; i <= n; ++i) {
            scanf("%d", &age[i]);
            sum += age[i];
        }
        while (m--) {
            int u, v;
            scanf("%d%d", &u, &v);
            if (u == v) {
                continue;
            }
            add_sub(u, 0, v, 0);
            if ((age[u] * n >= sum) == (age[v] * n >= sum)) {
                add_sub(u, 1, v, 1);
            }
        }
        if (!twoSAT()) {
            puts("No solution.");
            continue;
        }
        for (int i = 1; i <= n; ++i) {
            if (vis[i + n]) {
                puts("C");
            } else if (age[i] * n >= sum) {
                puts("A");
            } else {
                puts("B");
            }
        }
    }
    return 0;
}
```

## Now or Later

```cpp
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <cctype>
#include <vector>
#include <cmath>
#include <queue>
#include <stack>
using namespace std;
#define ll long long


const int N = 1e4 + 10;

int dfn[N],low[N],c[N],cnt,num,n;
bool ins[N];

stack <int> s;
vector <int> G[N];

void tarjan(int x) {
    dfn[x] = low[x] = ++cnt;
    s.push(x);ins[x] = 1;
    for (int i = 0;i < G[x].size();++i) {
        int v = G[x][i];
        if (!dfn[v]) {
            tarjan(v);
            low[x] = min(low[x],low[v]);
        } else if (ins[v]) {
            low[x] = min(low[x],dfn[v]);
        }
    }
    if (dfn[x] == low[x]) {
        int y;c[x] = ++num;
        do {
            y = s.top();s.pop();
            c[y] = num;ins[y] = 0;
        } while (x != y);
    }
}

int a[N][2];

bool check(int cap) {
    for (int i = 1;i <= 2 * n;++i) G[i].clear();
    memset(dfn,0,sizeof dfn);
    memset(low,0,sizeof low);
    memset(c,0,sizeof c);
    memset(ins,0,sizeof ins);
    cnt = num = 0;
    for (int i = 1;i <= n;++i) {
        for (int qwq = 0;qwq <= 1;++qwq) {
            for (int j = i + 1;j <= n;++j) {
                for (int qaq = 0;qaq <= 1;++qaq) {
                    if (abs(a[i][qwq] - a[j][qaq]) < cap) {
                        G[i + n * qwq].push_back(j + n * (qaq ^ 1));
                        G[j + n * qaq].push_back(i + n * (qwq ^ 1));
                    }
                }
            }
        }
    }//重新建图的过程
    for (int i = 1;i <= 2 * n;++i) if (!dfn[i]) tarjan(i);
    for (int i = 1;i <= n;++i) {
        if (c[i] == c[i+n]) return 0;
    }
    return 1;
}

void solve() {
    memset(ins,0,sizeof ins);
    tot = 0;
    int l = 0,r = 0,ans = 0;
    for (int i = 1;i <= n;++i) scanf("%d %d\n",&a[i][0],&a[i][1]),r = max(r,a[i][1]);
    while (l < r - 1) {
        int mid = (l + r) >> 1;
        if (check(mid)) l = mid;
        else r = mid;
    }
    cout << l << endl;
    return ;
}

signed main() {
    while (scanf("%d\n",&n) != EOF) solve();
    return 0;
}
```

## Grass Cownoisseur

```cpp
#include<cstdio>
#include<iostream>
#include<cstring>
#include<stack>
using namespace std;
struct point{
    int to,next;
}edge[200000],edge1[200000][3];
int cnt,n,m,ans,t,tot;
int head[200000],head1[200000][3],d[200000][3],dfn[200000],low[200000],v[200000];
int node[200000],f[200000][3],size[200000],q[200000];
stack<int>s;
void add(int u,int v)
{
  edge[++cnt].to=v;
  edge[cnt].next=head[u];
  head[u]=cnt;
}
void add1(int u,int v,int ch)
//ch为1的时候表示原本的缩点后的图，2的时候表示所有边都已反向过的图
{
   if (ch==1) cnt++;
   d[v][ch]++;
   edge1[cnt][ch].to=v;
   edge1[cnt][ch].next=head1[u][ch];
   head1[u][ch]=cnt;
}
void tarjan(int u)
{
  dfn[u]=low[u]=++t;//赋初值
  s.push(u);//放入栈中
  for (int i=head[u];i;i=edge[i].next)
      if (dfn[edge[i].to]==0)
//如果说这个点的dfn为0的话就表示这个点还未被搜索到过
      {
        tarjan(edge[i].to);
        low[u]=min(low[u],low[edge[i].to]);
    }
    else if (!v[edge[i].to])
//这个点没有出栈的话
     low[u]=min(low[u],dfn[edge[i].to]);
   if (dfn[u]==low[u])//如果这是一个强连通分量的根的话
   {
        tot++;//强连通分量数量加1，及缩点后的点数加1
        int now=0;
        while (now!=u)
        {
          now=s.top();//弹出栈
          s.pop();
       v[now]=true;//这个点已经出栈了
       node[now]=tot;//这个点属于哪一块强连通分量
       size[tot]++;    //这个强连通分量的大小加1
     }
   }
}
void find(int ch)
{
  f[node[1]][ch]=size[node[1]];
//ch依旧表示路是否被反向
//f[i][ch]数组则表示在当前的图，1到达[i]的路径上点权和最大为多少
  int ta=0;
//当前队列的尾巴
  for (int i=1;i<=tot;i++)
  if (d[i][ch]==0) q[++ta]=i;//如果说入度为0，加入队列
  while  (ta>0)//拓扑+dp就不详细解释了
  {
      int p=q[ta--];
      for (int i=head1[p][ch];i;i=edge1[i][ch].next)
      {
        f[edge1[i][ch].to][ch]=max(f[edge1[i][ch].to][ch],f[p][ch]+size[edge1[i][ch].to]);
      if (--d[edge1[i][ch].to][ch]==0) q[++ta]=edge1[i][ch].to;
    }
  }
}
int main()
{
  int x,y;
  scanf("%d%d",&n,&m);
  for (int i=1;i<=m;i++)
  {
      scanf("%d%d",&x,&y);
      add(x,y);//连接未缩点之前的边
  }
  t=0;
  for (int i=1;i<=n;i++) if (!v[i]) tarjan(i);
//用tarjan求强连通分量并进行缩点
//v[i]表示这个点有无出栈，如果为false的话，说明这个点还未被访问或者说还在栈中
 cnt=0;
  for (int i=1;i<=n;i++)
   for (int j=head[i];j;j=edge[j].next)
    if (node[i]!=node[edge[j].to])//此语句避免自环
    {
      add1(node[i],node[edge[j].to],1);//连边，之后搜索出的是1能到达的点
      add1(node[edge[j].to],node[i],2);//我们反向所有的边，这样搜索出来1能到达的点，其实是能到达1的点，这样就避免了构造n+1这个点
    }
  memset(f,0xef,sizeof f);//清空为一个很大的负数
  find(1);find(2);//搜索
  ans=size[node[1]];//开始答案的大小为1所在的强连通分量的大小
  for (int i=1;i<=n;i++)
   for (int j=head[i];j;j=edge[j].next)
//这里我们枚举所有未缩点前的边（当然这个取决于你自己）
//边的方向本来是node[i]——>node[edge[j].to]
//我们把它们反向，f[node[edge[j].to]][1]表示从点1到node[edge[j].to]，最大点权和为多少
//f[node[i]][2]表示从node[i]到点1最大点权和为多少
//这样边反向后，我们就可以成功的构建一条
//不过因为f[node[edge[j].to]][1]和f[node[i]][2]都包含了1所在的强连通分量的点权，所以我们需要减去一个size[node[1]]。
//1——>node[edge[j].to]——>node[i]——>1的路径了
    if (node[i]!=node[edge[j].to])
    ans=max(ans,f[node[edge[j].to]][1]+f[node[i]][2]-size[node[1]]);
   printf("%d\n",ans);
   return 0;
}
```

## Catowice City

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N = 2e6+9;
vector<int> v[N],w[N],scc[N];
int f[N], g[N], Sc[N];
stack<int> S;
void dfs(int x){
    f[x] = 1;
    for(int y : v[x]){
        if(!f[y]) dfs(y);
    }
    S.push(x);
}
int c= 0;
void ufs(int x){
    g[x] = 1;
    scc[c].push_back(x);
    Sc[x] = c;
    for(int y : w[x])
        if(!g[y]) ufs(y);
}
void solve(){
    c= 0;
    int n, m;
    scanf("%d%d", &n, &m);
    for(int i =1 ; i <= m; i++){
        int a, b;
        scanf("%d %d",&a, &b);
        v[a].push_back(b+n);
        w[b+n].push_back(a);
    }
    for(int i = 1; i <= n; i++)
        v[i+n].push_back(i),
        w[i].push_back(i+n);
    for(int i = 1; i <= 2*n; i++){
         if(f[i] == 0) dfs(i);
    }
    while(S.size()){
        int x = S.top();
        S.pop();
        if(g[x]) continue;
        c++;
        ufs(x);
    }
    for(int i = 1; i <= c; i++){
        int fl = 1,a=0,b=0;
        for(int j : scc[i]){
            if(j <= n) a++;
            else b++;
            for(int x : v[j])
                if(Sc[x] != i) fl = 0;
        }
        if(!fl || a != b || scc[i].size() == 2*n || scc[i].size()==0) continue;
        cout<<"Yes\n";
        printf("%d %d\n",scc[i].size()/2,n - scc[i].size()/2);
        for(int j = 1; j <= n; j++)
            if(Sc[j] == i) printf("%d ", j);
        cout<<endl;
        for(int j = 1; j <= n; j++)
            if(Sc[j+n] != i) printf("%d ", j);
        cout<<endl;

    for(int i = 1; i <= 2*n; i++)
        v[i].clear(), w[i].clear(), scc[i].clear(),f[i] =0,g[i] = 0,Sc[i] =0;
        return;
    }
    cout<<"No"<<endl;
    for(int i = 1; i <= 2*n; i++)
        v[i].clear(), w[i].clear(), scc[i].clear(),f[i] =0,g[i] = 0,Sc[i] =0;
}
main(){
    int T;
    cin >> T;
    while(T--){
        solve();
    }

}
```

## Festival

```cpp
#include<cstdio>
#include<iostream>
#include<cstring>
using namespace std;
const int N = 605;
struct edge {
	int next,to;
}a[N * 100];
int head[N],dis[N][N],maxn[N],n,m1,m2,a_size = 1,ans = 0;
inline void add(int u,int v) {
	a[++a_size] = (edge){head[u],v};
	head[u] = a_size;
}
int dfn[N],low[N],sta[N],c[N],top = 0,cnt = 0,num = 0;
bool ins[N];
void tarjan(int x) {
	dfn[x] = low[x] = ++num;
	sta[++top] = x,ins[x] = true;
	for(int i = head[x]; i; i = a[i].next) {
		int y = a[i].to;
		if(!dfn[y]) {
			tarjan(y);
			low[x] = min(low[x],low[y]);
		}
		else if(ins[y])
			low[x] = min(low[x],dfn[y]);
	}
	if(dfn[x] == low[x]) {
		cnt++; int y; do {
			y = sta[top--]; ins[y] = false;
			c[y] = cnt;
		}while(y != x);
	}
}
inline int read() {
	int x = 0,flag = 1;
	char ch = getchar();
	while(ch < '0' || ch > '9'){if(ch == '-')flag = -1;ch = getchar();}
	while(ch >='0' && ch <='9'){x = (x << 3) + (x << 1) + ch - 48;ch = getchar();}
	return x * flag;
}
int main() {
	memset(dis,0x3f,sizeof(dis));
	n = read(),m1 = read(),m2 = read();
	for(int i = 1; i <= n; i++) dis[i][i] = 0;
	for(int i = 1; i <= m1; i++) {
		int u = read(),v = read();
		add(u,v); add(v,u);
		dis[u][v] = min(dis[u][v],-1);
		dis[v][u] = min(dis[v][u],1);
	}
	for(int i = 1; i <= m2; i++) {
		int u = read(),v = read();
		add(u,v);
		dis[u][v] = min(dis[u][v],0);
	}
	for(int i = 1; i <= n; i++)
		if(!dfn[i]) tarjan(i);
	for(int k = 1; k <= n; k++) {
		for(int i = 1; i <= n; i++) {
			if(c[i] != c[k]) continue;
			for(int j = 1; j <= n; j++) {
				if(c[j] != c[i]) continue;
				dis[i][j] = min(dis[i][j],dis[i][k] + dis[k][j]);
			}
		}
	}
	for(int i = 1; i <= n; i++)
		if(dis[i][i]) {
			puts("NIE");
			return 0;
		}
	for(int i = 1; i <= n; i++) {
		for(int j = 1; j <= n; j++) {
			if(c[j] != c[i]) continue;
			maxn[c[i]] = max(maxn[c[i]],dis[i][j]);
		}
	}
	for(int i = 1; i <= cnt; i++) ans += maxn[i] + 1;
	printf("%d",ans);
	return 0;
}
```
