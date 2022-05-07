| POJ2828-Buy Tickets                    | 提高  | http://poj.org/problem?id=2828           |
| -------------------------------------- | ----- | ---------------------------------------- |
| Ray,Pass me the Dishes, UVa1400        | 提高  | https://www.luogu.com.cn/problem/UVA1400 |
| POJ3468 A Simple Problem with Integers | 提高  | http://poj.org/problem?id=3468           |
| USACO13Jan G, P3. Seating              | 提高  | https://www.luogu.com.cn/problem/P3071   |
| 维护序列(AHOI 2009)                    | 提高  | https://www.luogu.com.cn/problem/P2023   |
| USACO 2013 Dec G. Optimal Milking      | 提高+ | https://www.luogu.com.cn/problem/P3097   |

## Buy Tickets

```cpp
/**
 * @file poj/2828/segment_tree.cpp
 * @author Ruiming Guo (guoruiming@stu.scu.edu.cn)
 * @brief 有一个空的序列，有 n 个人要插队，第 i 个人有一个 id，
 * 这个人插入队列要排在 pos[i] 个人后面。
 * 最后从队首开始输出每个人的 id。
 *
 * 逆向思维，把插队问题改成出列问题，for i from n to 1,
 * 每个排在 pos[i] + 1 上的人出队，存下这个出队者的 id。
 * @version 1.0
 * @date 2022-05-06
 *
 * @copyright Copyright (c) 2022
 *
 **/

#include <iostream>
const int N = 200000 + 10;
int n;
int Pos[N], Val[N], Ans[N];

#define ls (rt << 1)
#define rs (rt << 1 | 1)
struct SegTree {
  int sum[4 * N];
  /**
   * @brief 使用子节点的信息更新父节点的信息
   *
   * @param rt 根节点
   **/
  void maintain(int rt) { sum[rt] = sum[ls] + sum[rs]; }
  /**
   * @brief 建立以 rt 为根节点，管理[l, r]的线段树
   *
   * @param rt 线段树的根节点
   * @param l 根节点 rt 管理的左短点
   * @param r 根节点 rt 管理的右端点
   **/
  void build(int rt, int l, int r) {
    if (l == r) {
      sum[rt] = 1;
      return;
    } else {
      int mid = l + r >> 1;
      build(ls, l, mid), build(rs, mid + 1, r);
      maintain(rt);
    }
  }
  /**
   * @brief 在以 rt 为根的树上，包括原序列上[l, r]这一段，记录 pos 位置上的人为
   *val
   *
   * @param rt 管理这一段的线段树的树根
   * @param l 这一段在原序列中的左端点
   * @param r 这一段在原序列中的右端点
   * @param pos 被改变的点的位置
   * @param val 该点被改变之后的值
   **/
  void update(int rt, int l, int r, int pos, int val) {
    if (l == r) {
      Ans[l] = val;
      sum[rt] = 0;  // 已经走到叶节点
      return;
    }
    int mid = l + r >> 1;
    if (sum[ls] >= pos)
      update(ls, l, mid, pos, val);
    else
      update(rs, mid + 1, r, pos - sum[ls], val);
    maintain(rt);  // 非叶子节点，需要调用
  }
} ST;
using namespace std;
int main() {
  ios::sync_with_stdio(0), cin.tie(0);
  while (cin >> n && n) {
    ST.build(1, 1, n);
    for (int i = 1; i <= n; ++i) {
      cin >> Pos[i] >> Val[i];
    }

    for (int i = n; i >= 1; --i) {
      ST.update(1, 1, n, Pos[i] + 1, Val[i]);
    }

    for (int i = 1; i <= n; ++i) cout << Ans[i] << ' ';
    cout << endl;
  }
}
```

## Ray, Pass me the Dishes!

用课件里的题解过的，不太会改

```cpp
#include <bits/stdc++.h>
using namespace std;
#define _for(i, a, b) for (int i = (a); i < (b); ++i)
#define _rep(i, a, b) for (int i = (a); i <= (b); ++i)
typedef long long LL;
typedef pair<int, int> Interval;
const int MAXN = 5e5 + 4;
LL SD[MAXN];
inline LL sum(int L, int R) {  // [L,R]
  // assert(L <= R);
  return SD[R] - SD[L - 1];
}
inline LL sum(const Interval& i) { return sum(i.first, i.second); }
inline Interval maxI(const Interval& i1, const Interval& i2) {
  LL s1 = sum(i1), s2 = sum(i2);
  if (s1 != s2) return s1 > s2 ? i1 : i2;
  return min(i1, i2);
}
struct MaxVal {
  int pfx, sfx;
  Interval sub;
};
struct IntervalTree {
  MaxVal Nodes[MAXN * 2];
  int qL, qR, N;
  void build(int N) {  // [1, N]
    this->N = N;
    build(1, N, 1);
  }
  void build(int L, int R, int O) {
    assert(L <= R);
    assert(O > 0);
    if (L == R) {
      Nodes[O] = {L, L, make_pair(L, L)};
      return;
    }
    int M = (L + R) / 2, lc = 2 * O, rc = 2 * O + 1;
    build(L, M, lc), build(M + 1, R, rc);
    const MaxVal &nl = Nodes[lc], &nr = Nodes[rc];
    MaxVal& no = Nodes[O];
    no.pfx = sum(L, nl.pfx) >= sum(L, nr.pfx) ? nl.pfx : nr.pfx;
    no.sfx = sum(nl.sfx, R) >= sum(nr.sfx, R) ? nl.sfx : nr.sfx;
    no.sub = maxI(nl.sub, nr.sub);
    no.sub = maxI(no.sub, make_pair(nl.sfx, nr.pfx));
  }
  Interval query(int l, int r) {  // max sub [a,b] in [l, r]
    assert(l <= r);
    qL = l, qR = r;
    return _query(1, N, 1);
  }

  Interval _query(const int L, const int R, const int O) {
    if (qL <= L && R <= qR) return Nodes[O].sub;
    int M = (L + R) / 2, lc = O * 2, rc = 2 * O + 1;
    if (qR <= M) return _query(L, M, lc);
    if (qL > M) return _query(M + 1, R, rc);
    Interval ans = make_pair(_querySfx(L, M, lc), _queryPfx(M + 1, R, rc));
    ans = maxI(ans, maxI(_query(L, M, lc), _query(M + 1, R, rc)));
    return ans;
  }
  int _queryPfx(const int L, const int R, const int O) {
    if (qL <= L && R <= qR) return Nodes[O].pfx;
    int M = (L + R) / 2, lc = 2 * O, rc = 2 * O + 1;
    if (qR <= M) return _queryPfx(L, M, lc);
    if (qL > M) return _queryPfx(M + 1, R, rc);
    int m1 = _queryPfx(L, M, lc), m2 = _queryPfx(M + 1, R, rc);
    return sum(L, m1) >= sum(L, m2) ? m1 : m2;
  }
  int _querySfx(const int L, const int R, const int O) {
    if (qL <= L && R <= qR) return Nodes[O].sfx;
    int M = (L + R) / 2, lc = O * 2, rc = 2 * O + 1;
    if (qR <= M) return _querySfx(L, M, lc);
    if (qL > M) return _querySfx(M + 1, R, rc);
    int m1 = _querySfx(L, M, lc), m2 = _querySfx(M + 1, R, rc);
    return sum(m1, R) >= sum(m2, R) ? m1 : m2;
  }
};
IntervalTree tree;
int main() {
  ios::sync_with_stdio(false), cin.tie(0);
  SD[0] = 0;
  for (int t = 1, d, a, b, N, M; cin >> N >> M; t++) {
    _rep(i, 1, N) cin >> d, SD[i] = SD[i - 1] + d;
    tree.build(N);
    printf("Case %d:\n", t);
    _rep(i, 1, M) {
      cin >> a >> b;
      Interval ans = tree.query(a, b);
      printf("%d %d\n", ans.first, ans.second);
    }
  }
  return 0;
}
```

## A Simple Problem with Integers

```cpp
/**
 * @file poj/3468
 * @author Ruiming Guo (guoruiming@stu.scu.edu.cn)
 * @brief 区间修改，区间查询，带懒标记
 * @see https://www.acwing.com/problem/content/244/
 * @version 1.0
 * @date 2022-05-06
 *
 * @copyright Copyright (c) 2022
 *
 **/
#include <cstdio>
using namespace std;
const int N = 100000 + 10;
#define mid (l + r >> 1)
#define ls (rt << 1)
#define rs (rt << 1 | 1)
typedef long long ll;
struct SegTree {
  ll sum[N * 4],  // sum[rt] 表示以 rt 为根的树管辖范围之和
      lazy_add[N * 4];  // lazy_add[rt] 表示 rt 管辖的范围中每个元素都增加了多少
  /**
   * @brief 以 rt 为根节点的线段树用子节点的信息更新自己
   *
   * @param rt 根节点
   **/
  void maintain(int rt) { sum[rt] = sum[ls] + sum[rs]; }
  /**
   * @brief 下放延迟标记，把lazy_add下放至两棵子树，同时更新子树的sum
   *
   * @param rt 根节点
   * @param l rt 管辖范围的左边界
   * @param r rt 管辖范围的右边界
   **/
  void pushdown(int rt, int l, int r) {
    ll &a = lazy_add[rt];
    if (a == 0) return;
    lazy_add[ls] += a, lazy_add[rs] += a;  // 下面压力来到了左右子节点上
    sum[ls] += a * (mid - l + 1), sum[rs] += a * (r - mid);
    a = 0;  // 清掉 rt 的标记
  }
  void build(int rt, int l, int r) {
    lazy_add[rt] = 0;
    if (l == r) {
      // 建树同时读入，因为建树时建到子节点的顺序是从 1 到 n 的
      scanf("%lld", &sum[rt]);
      return;
    }
    build(ls, l, mid), build(rs, mid + 1, r);
    maintain(rt);
  }
  /**
   * @brief 改
   *
   * @param rt 修改的根节点
   * @param val 被改成的值
   * @param ql 询问区间的左端点
   * @param qr 询问区间的右端点
   * @param l 该根节点控制区间的左端点
   * @param r 该根节点控制区间的右端点
   **/
  void update(int rt, ll val, int ql, int qr, int l, int r) {
    if (ql <= l && r <= qr) {  // rt 区间完全被询问 [L, R] 覆盖
                               // 进行整个区间的更新
      lazy_add[rt] += val;
      sum[rt] += val * (r - l + 1);
      return;
    }
    // 非完整区间更新
    pushdown(rt, l, r);
    if (ql <= mid) {
      update(ls, val, ql, qr, l, mid);
    }
    if (qr >= mid + 1) {
      update(rs, val, ql, qr, mid + 1, r);
    }
    maintain(rt);
  }
  /**
   * @brief 查
   *
   * @param rt 修改的根节点
   * @param ql 询问区间的左端点
   * @param qr 询问区间的右端点
   * @param l 该根节点控制区间的左端点
   * @param r 该根节点控制区间的右端点
   **/
  ll query(int rt, int ql, int qr, int l, int r) {
    if (ql <= l && r <= qr) return sum[rt];  // rt 区间完全被询问 [L, R] 覆盖
    pushdown(rt, l, r);
    ll ans = 0;
    if (ql <= mid) ans += query(ls, ql, qr, l, mid);
    if (qr >= mid + 1) ans += query(rs, ql, qr, mid + 1, r);
    return ans;
  }
} ST;
int main() {
  int n, m;
  scanf("%d%d", &n, &m);
  ST.build(1, 1, n);
  char s[2];
  for (int a, b, c; m--;) {
    scanf("%s%d%d", s, &a, &b);
    if (*s == 'Q') {
      printf("%lld\n", ST.query(1, a, b, 1, n));
    } else {
      scanf("%d", &c);
      ST.update(1, c, a, b, 1, n);
    }
  }
}
```

## Seating

自己做没做出来，怎么调都没有调对……把课件里代码敲了一遍

```cpp
#include <bits/stdc++.h>
const int NN = 5e5 + 4;
using namespace std;
int opv[NN * 4], pfx[NN * 4], sfx[NN * 4], len[NN * 4];
void maintain(int o, int lLen, int rLen) {
  int lc = 2 * o, rc = 2 * o + 1;
  pfx[o] = pfx[lc] == lLen ? lLen + pfx[rc] : pfx[lc];
  sfx[o] = sfx[rc] == rLen ? rLen + sfx[lc] : sfx[rc];
  len[o] = max(max(len[lc], len[rc]), pfx[rc] + sfx[lc]);
}
void push_down(int o, int lLen, int rLen) {
  int &op = opv[o];
  if (!op || !lLen || !rLen) return;
  int lc = 2 * o, rc = 2 * o + 1;
  len[lc] = pfx[lc] = sfx[lc] = (op == 2 ? lLen : 0);
  len[rc] = pfx[rc] = sfx[rc] = (op == 2 ? rLen : 0);
  opv[lc] = opv[rc] = op, op = 0;
}
int query(int o, int L, int R, int v) {
  // 在 o[L,R]中查找长度为q的连续0区间
  int lc = 2 * o, rc = 2 * o + 1, m = L + (R - L) / 2;
  push_down(o, m - L + 1, R - m);
  if (len[lc] >= v) return query(lc, L, m, v);         // 去左半边找
  if (pfx[rc] + sfx[lc] >= v) return m - sfx[lc] + 1;  // 横跨中间线
  return query(rc, m + 1, R, v);                       // 去右边找
}

void update(int qL, int qR, int op, int o, int L, int R) {
  int lc = 2 * o, rc = 2 * o + 1, m = L + (R - L) / 2;
  if (qL <= L && qR >= R) {
    len[o] = pfx[o] = sfx[o] = (op == 2 ? R - L + 1 : 0);
    opv[o] = op;
    return;
  }
  push_down(o, m - L + 1, R - m);
  if (qL <= m) update(qL, qR, op, lc, L, m);
  if (qR > m) update(qL, qR, op, rc, m + 1, R);
  maintain(o, m - L + 1, R - m);
}
void build(int o, int L, int R) {
  int lc = 2 * o, rc = 2 * o + 1, m = L + (R - L) / 2;
  len[o] = pfx[o] = sfx[o] = R - L + 1;
  if (L < R) build(lc, L, m), build(rc, m + 1, R);
}
int main() {
  ios::sync_with_stdio(false), cin.tie(0);
  int n, m, ans = 0;
  char op;
  cin >> n >> m, build(1, 1, n);
  for (int i = 0, a, b; i < m; i++) {
    cin >> op >> a;
    if (op == 'A') {
      if (len[1] < a)
        ans++;
      else {
        int x = query(1, 1, n, a);
        update(x, x + a - 1, 1, 1, 1, n);
      }
    } else
      cin >> b, update(a, b, 2, 1, 1, n);
  }
  cout << ans << endl;
}

```

## 维护序列

```cpp
/**
 * @file luogu/2023/main.cpp
 * @author Ruiming Guo (guoruiming@stu.scu.edu.cn)
 * @brief 区间加，区间乘，询问区间和
 *
 * 带懒标记的线段树
 *
 * 注意：因为乘的运算级别比加高，所以在做加法是不用管乘法，在做乘法时要管加法。
 *
 * @version 1.0
 * @date 2022-05-07
 *
 * @copyright Copyright (c) 2022
 *
 **/

#include <bits/stdc++.h>
using namespace std;
const int N = 200007;
typedef long long ll;
int n, q, L, R, op;
ll k, P, a[N];
struct SegTree {
#define ls (rt << 1)
#define rs (rt << 1 | 1)
#define mid (l + r >> 1)
  ll sum[N << 2], mul[N << 2], add[N << 2];
  void pushdown(int rt, int l, int r) {
    if (mul[rt] == 1 && add == 0) {
      return;  // 没有标记
    }
    if (l != r) {
      mul[ls] = mul[ls] * mul[rt] % P;
      mul[rs] = mul[rs] * mul[rt] % P;

      add[ls] = (add[ls] * mul[rt] % P + add[rt]) % P;
      add[rs] = (add[rs] * mul[rt] % P + add[rt]) % P;
    }
    sum[rt] = (sum[rt] * mul[rt] % P + add[rt] * (r - l + 1) % P) % P;

    mul[rt] = 1, add[rt] = 0;
  }
  void pushup(int rt) { sum[rt] = sum[ls] + sum[rs]; }
  void build(int rt, int l, int r) {
    mul[rt] = 1, add[rt] = 0;
    if (l == r) {
      sum[rt] = a[l];
      return;
    }
    build(ls, l, mid), build(rs, mid + 1, r);
    pushup(rt);
  }
  ll query_sum(int rt, int l, int r) {
    pushdown(rt, l, r);
    if (L <= l && r <= R) {
      return sum[rt];
    }
    ll ret = 0;
    if (L <= mid) ret = (ret + query_sum(ls, l, mid)) % P;
    if (R >= mid + 1) ret = (ret + query_sum(rs, mid + 1, r)) % P;
    return ret;
  }
  void range_add(int rt, int l, int r) {
    pushdown(rt, l, r);
    if (L <= l && r <= R) {
      add[rt] = (add[rt] + k) % P;
      return;
    }
    if (L <= mid) range_add(ls, l, mid);
    if (R >= mid + 1) range_add(rs, mid + 1, r);
    pushdown(ls, l, mid), pushdown(rs, mid + 1, r);
    pushup(rt);
  }
  void range_mul(int rt, int l, int r) {
    pushdown(rt, l, r);
    if (L <= l && r <= R) {
      // 加法和乘法都要乘一个 k
      mul[rt] = mul[rt] * k % P;
      add[rt] = add[rt] * k % P;
      return;
    }
    if (L <= mid) range_mul(ls, l, mid);
    if (R >= mid + 1) range_mul(rs, mid + 1, r);
    pushdown(ls, l, mid), pushdown(rs, mid + 1, r);
    pushup(rt);
  }
#undef mid
#undef ls
#undef rs
} ST;
int main() {
  cin >> n >> P;
  for (int i = 1; i <= n; ++i) cin >> a[i];
  ST.build(1, 1, n);
  int tc;
  cin >> tc;
  while (tc--) {
    cin >> op;
    if (op == 1) {
      cin >> L >> R >> k;
      ST.range_mul(1, 1, n);
    } else if (op == 2) {
      cin >> L >> R >> k;
      ST.range_add(1, 1, n);
    } else {
      cin >> L >> R;
      cout << ST.query_sum(1, 1, n) << endl;
    }
  }
}
```

## Optimal Milking

看不太懂了，好像是要维护总体、不含左端点、不含右端点、不含两个端点四种情况

```cpp
#include <bits/stdc++.h>
#define N 50010
#define lson (o << 1)
#define rson (o << 1 | 1)
using namespace std;
typedef long long ll;
int f[N << 2][4], n, m;
ll ans = 0;
inline int read() {
  int f = 1, x = 0;
  char ch;
  do {
    ch = getchar();
    if (ch == '-') f = -1;
  } while (ch < '0' || ch > '9');
  do {
    x = x * 10 + ch - '0';
    ch = getchar();
  } while (ch >= '0' && ch <= '9');
  return f * x;
}
inline int max(int a, int b, int c) { return max(max(a, b), c); }
inline int max(int a, int b, int c, int d) { return max(max(a, b), max(c, d)); }
inline void pushup(int o) {
  for (int i = 0; i < 4; i++)
    f[o][i] = max(f[lson][i & 2] + f[rson][2 + (i & 1)],
                  f[lson][(i & 2) + 0] + f[rson][i & 1],
                  f[lson][(i & 2) + 1] + f[rson][i & 1]);
}
void build(int o, int l, int r) {
  if (l == r) {
    f[o][3] = read();
    return;
  }
  int mid = (l + r) >> 1;
  build(lson, l, mid);
  build(rson, mid + 1, r);
  pushup(o);
}
void change(int o, int l, int r, int q, int v) {
  if (l == r) {
    f[o][3] = v;
    return;
  }
  int mid = (l + r) >> 1;
  if (q <= mid)
    change(lson, l, mid, q, v);
  else
    change(rson, mid + 1, r, q, v);
  pushup(o);
}
int main() {
  n = read();
  m = read();
  build(1, 1, n);
  while (m--) {
    int x = read(), y = read();
    change(1, 1, n, x, y);
    ans += max(f[1][0], f[1][1], f[1][2], f[1][3]);
  }
  cout << ans << endl;
}
```
