# 20220320-DFS 序&欧拉序列

| 投入时间 |          |                                             |       |                                                                                   |
| -------- | -------- | ------------------------------------------- | ----- | --------------------------------------------------------------------------------- |
| < 0.5h   | 时间戳   | 祖孙询问(LOJ10135)                          | 提高  | https://loj.ac/p/10135                                                            |
| < 0.75h  | 时间戳   | AtCoder abc202 E Count Descendants          | 提高+ | https://atcoder.jp/contests/abc202/tasks/abc202_e                                 |
| <1h      | 欧拉序列 | USACO 2019 Feb G. Cow Land                  | 提高+ | [https://www.luogu.com.cn/problem/P6098 ](https://www.luogu.com.cn/problem/P6098) |
| <2h      | 欧拉序列 | USACO 2019 Dec G. Milk Visits(**离线算法**) | 提高+ | https://www.luogu.com.cn/problem/P5838                                            |

| 投入时间 |          |                                                 |      |                                              |
| -------- | -------- | ----------------------------------------------- | ---- | -------------------------------------------- |
| <1h      | 欧拉序列 | USACO 2019 Dec G. Milk Visits(**禁止离线算法**) | NOI- | https://www.luogu.com.cn/problem/P5838       |
| <1h      | 时间戳   | [SDOI2015]寻宝游戏                              | NOI- | https://www.luogu.com.cn/problem/P3320       |
| <1h      | 时间戳   | USACO 2019 Dec P. Bessie's Snow Cow             | NOI- | https://www.luogu.com.cn/problem/P5852       |
| <2h      | 欧拉序列 | IOI2009 – Regions                               | NOI  | https://www.luogu.com.cn/problem/P5901       |
| <1.5h    | 时间戳   | CF838B Diverging Directions                     | NOI- | https://codeforces.com/contest/838/problem/B |

## Milking Visits

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef vector<int> vi;
const int NN = 1e5 + 4;
int T[NN];
struct LCA {
  static const int HH = 18;
  int timer = 0, Tin[NN], fa[HH][NN], Dep[NN];
  vi G[NN], W[NN];
  vector<pair<int, int>> X[NN];
  void addEdge(int u, int v) { G[u].push_back(v), G[v].push_back(u); }
  void dfs(int u, int f) {
    auto &w = W[T[u]];
    auto &x = X[T[u]];
    w.push_back(u), x.push_back({Tin[u] = timer++, u});
    fa[0][u] = f, Dep[u] = Dep[f] + 1;
    for (int k = 1; k < HH; ++k) fa[k][u] = fa[k - 1][fa[k - 1][u]];
    for (int v : G[u])
      if (v != f) dfs(v, u);
    w.pop_back(), x.push_back({timer, !w.empty() ? w.back() : 0});
  }
  int lca(int u, int v) {
    if (Dep[u] < Dep[v]) swap(u, v);
    for (int k = HH - 1; k >= 0; --k)
      if ((Dep[u] - Dep[v]) & (1 << k)) u = fa[k][u];
    for (int k = HH - 1; k >= 0; --k)
      if (fa[k][u] != fa[k][v]) u = fa[k][u], v = fa[k][v];
    return u == v ? u : fa[0][u];
  }
  int last(int u, int c) {
    auto &x = X[c];
    auto it = upper_bound(begin(x), end(x), make_pair(Tin[u], NN + 10));
    return it == begin(x) ? 0 : prev(it)->second;
  }
} L;
int solve(int u, int v, int c) {
  int d = L.lca(u, v), a = L.last(u, c);
  if (a && L.Dep[a] >= L.Dep[d]) return 1;
  a = L.last(v, c);
  return a && L.Dep[a] >= L.Dep[d];
}
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  int N, M;
  cin >> N >> M;
  for (int i = 1; i <= N; ++i) cin >> T[i];
  for (int i = 1, u, v; i < N; ++i) cin >> u >> v, L.addEdge(u, v);
  L.dfs(1, 0);
  for (int i = 0, A, B, C; i < M; ++i)
    cin >> A >> B >> C, cout << solve(A, B, C);
  cout << endl;
}
```

## 寻宝游戏

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
const int NN = 1e5 + 4;
struct Edge {
  int v, w;
};
LL Dis[NN];
int N, L, M, UP[NN][18], Tin[NN], Tout[NN], Timer;
vector<Edge> G[NN];
void dfs(int u, int fa) {
  UP[u][0] = fa, Tin[u] = ++Timer;
  for (int i = 1; i <= L; ++i) UP[u][i] = UP[UP[u][i - 1]][i - 1];
  for (const Edge &e : G[u])
    if (e.v != fa) Dis[e.v] = Dis[u] + e.w, dfs(e.v, u);
  Tout[u] = ++Timer;
}
bool isAncestor(int u, int v) { return Tin[u] <= Tin[v] && Tout[u] >= Tout[v]; }
int lca(int u, int v) {
  if (isAncestor(v, u)) return v;
  if (isAncestor(u, v)) return u;
  for (int i = L; i >= 0; --i)
    if (!isAncestor(UP[u][i], v)) u = UP[u][i];
  return UP[u][0];
}
LL dist(int u, int v) { return (LL)Dis[u] + Dis[v] - Dis[lca(u, v)] * 2ll; }
struct NodeComp {
  bool operator()(int u, int v) const { return Tin[u] < Tin[v]; }
};
set<int, NodeComp> S;
typedef set<int, NodeComp>::iterator SIT;
inline SIT leftOf(SIT it) {
  if (it == S.begin()) return --S.end();
  return --it;
}
inline SIT rightOf(SIT it) {
  if (++it == S.end()) return S.begin();
  return it;
}
inline LL diff(int u) {
  SIT i = S.find(u), r = rightOf(i), l = leftOf(i);
  return dist(u, *l) + dist(u, *r) - dist(*l, *r);
}
int main() {
  ios::sync_with_stdio(false), cin.tie(0);
  cin >> N >> M, Timer = 0, L = ceil(log2(N));
  for (int i = 1, u, v, w; i <= N - 1; ++i)
    cin >> u >> v >> w, G[u].push_back({v, w}), G[v].push_back({u, w});
  dfs(1, 1);
  LL ans = 0;
  for (int i = 1, u; i <= M; ++i) {
    cin >> u;
    if (S.count(u))
      ans -= diff(u), S.erase(u);
    else
      S.insert(u), ans += diff(u);
    cout << ans << endl;
  }
  return 0;
}
```

## Bessie’s Snow Cow

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 1e5 + 10;
int n, q;
struct bit {
  int t[N];
  void add(int x, int v) {
    for (int i = x; i <= n; i += i & -i) t[i] += v;
  }
  int qry(int x) {
    int res = 0;
    for (int i = x; i; i -= i & -i) res += t[i];
    return res;
  }
} t1, t2;
struct edge {
  int to, nxt;
} e[N << 1];
int head[N], cnt;
void add(int u, int v) {
  e[++cnt] = (edge){v, head[u]};
  head[u] = cnt;
}
set<int> s[N];
set<int>::iterator it;
int dfn[N], siz[N], id[N], tim;
void dfs(int u, int fa) {
  dfn[u] = ++tim, id[tim] = u, siz[u] = 1;
  for (int i = head[u], v; i; i = e[i].nxt) {
    v = e[i].to;
    if (v == fa) continue;
    dfs(v, u);
    siz[u] += siz[v];
  }
}
int main() {
  cin >> n >> q;
  for (int i = 1, u, v; i < n; i++) {
    cin >> u >> v;
    add(u, v), add(v, u);
  }
  dfs(1, 0);
  for (int i = 1, opt, u, c; i <= q; i++) {
    cin >> opt >> u;
    if (opt == 1) {
      cin >> c;
      it = s[c].lower_bound(dfn[u]);
      if (it != s[c].begin() &&
          dfn[id[*prev(it)]] + siz[id[*prev(it)]] >= dfn[u] + siz[u])
        continue;
      while (it != s[c].end() && *it < dfn[u] + siz[u])
        t1.add(*it, -1), t1.add(*it + siz[id[*it]], 1),
            t2.add(*it, -siz[id[*it]]), s[c].erase(it++);
      s[c].insert(dfn[u]), t1.add(dfn[u], 1), t1.add(dfn[u] + siz[u], -1),
          t2.add(dfn[u], siz[u]);
    } else
      printf("%d\n", siz[u] * t1.qry(dfn[u]) + t2.qry(dfn[u] + siz[u] - 1) -
                         t2.qry(dfn[u]));
  }
  return 0;
}
```

## Bessie’s snow cow

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 1e5 + 10;
int n, q;
struct bit {
  int t[N];
  void add(int x, int v) {
    for (int i = x; i <= n; i += i & -i) t[i] += v;
  }
  int qry(int x) {
    int res = 0;
    for (int i = x; i; i -= i & -i) res += t[i];
    return res;
  }
} t1, t2;
struct edge {
  int to, nxt;
} e[N << 1];
int head[N], cnt;
void add(int u, int v) {
  e[++cnt] = (edge){v, head[u]};
  head[u] = cnt;
}
set<int> s[N];
set<int>::iterator it;
int dfn[N], siz[N], id[N], tim;
void dfs(int u, int fa) {
  dfn[u] = ++tim, id[tim] = u, siz[u] = 1;
  for (int i = head[u], v; i; i = e[i].nxt) {
    v = e[i].to;
    if (v == fa) continue;
    dfs(v, u);
    siz[u] += siz[v];
  }
}
int main() {
  cin >> n >> q;
  for (int i = 1, u, v; i < n; i++) {
    cin >> u >> v;
    add(u, v), add(v, u);
  }
  dfs(1, 0);
  for (int i = 1, opt, u, c; i <= q; i++) {
    cin >> opt >> u;
    if (opt == 1) {
      cin >> c;
      it = s[c].lower_bound(dfn[u]);
      if (it != s[c].begin() &&
          dfn[id[*prev(it)]] + siz[id[*prev(it)]] >= dfn[u] + siz[u])
        continue;
      while (it != s[c].end() && *it < dfn[u] + siz[u])
        t1.add(*it, -1), t1.add(*it + siz[id[*it]], 1),
            t2.add(*it, -siz[id[*it]]), s[c].erase(it++);
      s[c].insert(dfn[u]), t1.add(dfn[u], 1), t1.add(dfn[u] + siz[u], -1),
          t2.add(dfn[u], siz[u]);
    } else
      printf("%d\n", siz[u] * t1.qry(dfn[u]) + t2.qry(dfn[u] + siz[u] - 1) -
                         t2.qry(dfn[u]));
  }
  return 0;
}
```

## Regions

```cpp
// IOI 2009 Regions

#include <bits/stdc++.h>
using namespace std;
typedef vector<int> vi;
typedef pair<int, int> pi;
typedef long long ll;
struct Emp {
  int tin, reg;
  vi ch;  // 被指导的
};
struct Reg {
  vi ids;             // 其中的Employee
  vector<pi> ranges;  // {id, cnt}
  int cnt;
};
vector<Emp> ES;
vector<Reg> RS;
void dfs(int u, int &timer) {
  auto &r = RS[ES[u].reg];
  r.ids.push_back(timer++), r.ranges.push_back({timer, ++r.cnt});
  for (int v : ES[u].ch) dfs(v, timer);
  r.ranges.push_back({timer, --r.cnt});
}
ll query_by_id(const Reg &r1, const Reg &r2) {
  ll ans = 0;
  auto &rv = r1.ranges;
  for (int u : r2.ids) {
    auto it = lower_bound(begin(rv), end(rv), make_pair(u, INT_MAX));
    if (it != rv.begin()) ans += prev(it)->second;
  }
  return ans;
}
ll query_by_range(const Reg &r1, const Reg &r2) {
  ll ans = 0;
  const auto &rv = r2.ids;
  for (size_t i = 0; i + 1 < r1.ranges.size(); ++i) {
    int p1 = r1.ranges[i].first, p2 = r1.ranges[i + 1].first;
    auto it1 = lower_bound(begin(rv), end(rv), p1),
         it2 = lower_bound(begin(rv), end(rv), p2);
    ans += r1.ranges[i].second * (it2 - it1);
  }
  return ans;
}
ll solve(int r1, int r2) {
  static map<pi, ll> M;
  pi k(r1, r2);
  if (M.count(k)) return M[k];
  const Reg &reg1 = RS[r1], &reg2 = RS[r2];
  int s1 = reg1.ids.size(), s2 = reg2.ids.size();
  int tm1 = (log2(s2) + 2) * s1, tm2 = (log2(s1) + 2) * s2;
  ll ans = tm1 < tm2 ? query_by_range(reg1, reg2) : query_by_id(reg1, reg2);
  return M[k] = ans;
}
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  int N, R, Q, timer = 0;
  cin >> N >> R >> Q, ES.resize(N), RS.resize(R);
  cin >> ES[0].reg, --ES[0].reg;
  for (int i = 1, fa; i < N; ++i)
    cin >> fa >> ES[i].reg, --ES[i].reg, ES[fa - 1].ch.push_back(i);
  dfs(0, timer);
  for (int q = 0, r1, r2; q < Q; ++q)
    cin >> r1 >> r2, cout << solve(r1 - 1, r2 - 1) << endl;
  return 0;
}
```

## Diverging Directions

```cpp
#include <bits/stdc++.h>
using namespace std;
struct Edge {
  int v, w;
};
typedef vector<int> vi;
const int VV = 1e6 + 4, INF = 1e9;
vi G[VV];
vector<Edge> Es;
void addEdge(int u, int v, int w) {
  G[u].push_back(Es.size()), Es.push_back({v, w});
}
int Sz[VV], Pw[VV], timer;
int Pre[VV], B[VV], D[VV], Tin[VV], Tout[VV];
struct SegTree {
  int minv[VV], addv[VV];
  inline void maintain(int o) { minv[o] = min(minv[2 * o], minv[2 * o + 1]); }
  void build(int o, int L, int R) {
    if (L == R) {
      minv[o] = B[Pre[L]] + D[Pre[L]];
      return;
    }
    int lc = 2 * o, rc = 2 * o + 1, m = (L + R) / 2;
    build(lc, L, m), build(rc, m + 1, R), maintain(o);
  }
  inline void pushdown(int o) {
    int &a = addv[o], lc = 2 * o, rc = 2 * o + 1;
    if (a == 0) return;
    addv[lc] += a, addv[rc] += a, minv[lc] += a, minv[rc] += a;
    a = 0;
  }
  int querymin(int o, int L, int R, int ql, int qr) {
    int lc = 2 * o, rc = 2 * o + 1, m = (L + R) / 2;
    if (L > qr || R < ql) return INF;
    if (L >= ql && R <= qr) return minv[o];
    pushdown(o);
    return min(querymin(lc, L, m, ql, qr), querymin(rc, m + 1, R, ql, qr));
  }
  void add(int o, int L, int R, int ql, int qr, int v) {
    if (L > qr || R < ql) return;
    if (L >= ql && R <= qr) return (void)(minv[o] += v, addv[o] += v);
    int lc = 2 * o, rc = 2 * o + 1, m = (L + R) / 2;
    pushdown(o);
    add(lc, L, m, ql, qr, v), add(rc, m + 1, R, ql, qr, v);
    maintain(o);
  }
};
void dfs(int u, int fa) {
  Tin[u] = ++timer, Pre[timer] = u;
  for (int ei : G[u]) {
    const Edge &e = Es[ei];
    if (e.v == fa) continue;
    D[e.v] = D[u] + e.w, Pw[e.v] = e.w, dfs(e.v, u);
  }
  Tout[u] = timer;
}
SegTree S;
int N, M;
inline int getD(int x) { return S.querymin(1, 1, N, Tin[x], Tin[x]) - B[x]; }
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  cin >> N >> M;
  for (int i = 1, u, v, w; i < N; ++i) cin >> u >> v >> w, addEdge(u, v, w);
  vi E(2 * N);
  for (int i = N, u, v, w; i <= 2 * N - 2; ++i)
    cin >> u >> v >> w, B[u] = w, E[i] = u;
  dfs(1, 0), S.build(1, 1, N);
  while (M--) {
    int o, i, w, u, v;
    cin >> o;
    if (o == 1) {
      cin >> i >> w;
      if (i >= N)
        u = E[i], S.add(1, 1, N, Tin[u], Tin[u], w - B[u]), B[u] = w;
      else
        u = Es[i - 1].v, S.add(1, 1, N, Tin[u], Tout[u], w - Pw[u]), Pw[u] = w;
    } else {
      cin >> u >> v;
      if (Tin[u] <= Tin[v] && Tin[v] <= Tout[u])
        cout << getD(v) - getD(u) << endl;
      else
        cout << S.querymin(1, 1, N, Tin[u], Tout[u]) - getD(u) + getD(v)
             << endl;
    }
  }
}
```
