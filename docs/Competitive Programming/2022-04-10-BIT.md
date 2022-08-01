## Stars

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 32005;
int n, tree[N], sum[N], stars[N], x, y;
int lowbit(int i) { return i & -i; }
void update(int i, int x) {
  while (i < N) {
    tree[i] = tree[i] + x;
    i += lowbit(i);
  }
}
int query(int n) {
  int sum = 0;
  while (n > 0) {
    sum += tree[n];
    n -= lowbit(n);
  }
  return sum;
}
int main() {
  while (~scanf("%d", &n)) {
    memset(sum, 0, sizeof(sum));
    memset(tree, 0, sizeof(tree));
    memset(stars, 0, sizeof(stars));

    for (int i = 1; i <= n; ++i) {
      scanf("%d%d", &x, &y);
      update(x + 1, 1);
      int cnt = query(x + 1) - 1;
      stars[cnt]++;
    }
    for (int i = 0; i < n; ++i) printf("%d\n", stars[i]);
  }
}
```

## Ultra-QuickSort

```cpp
#include <bits/stdc++.h>
typedef long long ll;
using namespace std;
const int N = 5e5 + 10;
const int mod = 1e9 + 7;
struct node {
  int num, id;
  bool operator<(const node &b) const { return num < b.num; }
} a[N];
int c[N], b[N], n;
ll ans;
int inline lowbit(int x) { return x & -x; }
void update(int x, int k) {
  while (x <= n) {
    c[x] += k;
    x += lowbit(x);
  }
}
int query(int x) {
  int res = 0;
  while (x > 0) {
    res += c[x];
    x -= lowbit(x);
  }
  return res;
}
int main() {
  while (scanf("%d", &n) != EOF) {
    if (n == 0) break;
    memset(c, 0, sizeof(c));
    for (int i = 1; i <= n; ++i) {
      scanf("%d", &a[i].num);
      a[i].id = i;
    }
    sort(a + 1, a + n + 1);
    for (int i = 1; i <= n; ++i) b[a[i].id] = i;
    ll ans = 0;
    for (int i = 1; i <= n; ++i) {
      update(b[i], 1);
      ans += 1LL * i - query(b[i]);
    }
    cout << ans << endl;
  }
}
```

### Japan

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 1010;
ll mat[N][N], n, m;
int a[N * N], b[N * N];
int lowbit(int x) { return x & -x; }
void add(int x, int y, int d) {
  for (int i = x; i <= n; i += lowbit(i))
    for (int j = y; j <= m; j += lowbit(j)) mat[i][j] += d;
}
ll sum(int x, int y) {
  if (x <= 0 || y <= 0) return 0;
  ll ret = 0;
  for (int i = x; i > 0; i -= lowbit(i))
    for (int j = y; j > 0; j -= lowbit(j)) ret += mat[i][j];
  return ret;
}
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  int t, k, kase = 0;
  cin >> t;
  while (cin >> n >> m >> k) {
    memset(mat, 0, sizeof(mat));
    for (int i = 0; i < k; ++i) {
      cin >> a[i] >> b[i];
      add(a[i], b[i], 1);
    }
    ll ans = 0;
    for (int i = 0; i < k; ++i) {
      ans += sum(n, b[i] - 1) + sum(a[i] - 1, m) - sum(a[i], b[i] - 1) -
             sum(a[i] - 1, b[i]);
    }
    cout << "Test case " << ++kase << ": " << ans / 2 << endl;
  }
  return 0;
}
```

### Mega Inversions

```cpp
#include <bits/stdc++.h>

using namespace std;
using ll = long long;
const int MX = 2e5 + 5;

ll bit2[MX], bit1[MX];
int n;

void upd (int i, int val, ll ar[]) {
	for (; i <= n; i += i&(-i)) {
		ar[i] += val;
	}
}

ll query (int i, ll ar[]) {
	ll res = 0;
	for (; i; i -= i&(-i)) {
		res += ar[i];
	}
	return res;
}

int main () {
	cin >> n;
	vector<int> ar(n);
	for (int i = 0; i < n; i++) {
		cin >> ar[i];
	}

	ll sol = 0;
	for (int i = n - 1; i >= 0; i--) {
		sol += query(ar[i] - 1, bit2);
		ll uno = query(ar[i] - 1, bit1);
		upd(ar[i], 1, bit1);
		upd(ar[i], uno, bit2);
	}
	cout << sol << '\n';
}
```

### Haircut

```cpp
#include <bits/stdc++.h>
using namespace std;
#define int long long
const int N = 100100;
int n, a[N];
int s[N], ans;
struct bit {
  int tree[N];
  int lowbit(int x) { return x & -x; }
  void update(int x, int y) {
    while (x <= n) {
      tree[x] += y;
      x += lowbit(x);
    }
  }
  int query(int x) {
    int ret = 0;
    while (x) {
      ret += tree[x];
      x -= lowbit(x);
    }
    return ret;
  }
} t;
signed main() {
  scanf("%lld", &n);
  for (int i = 1; i <= n; ++i) {
    scanf("%lld", &a[i]);
    a[i]++;
  }
  for (int i = 1; i <= n; ++i) {
    int x = n - a[i] + 2;
    s[a[i]] += t.query(x - 1);
    t.update(x, 1);
  }
  printf("0\n");
  for (int i = 2; i <= n; ++i) {
    ans += s[i - 1];
    printf("%lld\n", ans);
  }
}
```

### 简单题

```cpp
#include <cstdio>
const int N = 100010;
int n, m, b[N];
inline void add(int i) {
  for (; i <= N; i += i & -i) b[i] ^= 1;
}
inline int query(int i) {
  int res = 0;
  for (; i; i -= i & -i) res ^= b[i];
  return res;
}
int main() {
  scanf("%d%d", &n, &m);
  while (m--) {
    int op, l, r;
    scanf("%d%d", &op, &l);
    if (op == 1)
      add(l), scanf("%d", &r), add(r + 1);
    else
      printf("%d\n", query(l));
  }
}
```

### Cows

```cpp
#include <algorithm>
#include <cstdio>
#include <cstring>

using namespace std;
const int N = 100010;
int c[N];
struct Node {
  int S, E;
  int index;
  bool operator<(const Node &b) const {
    if (E == b.E) return S < b.S;
    return E > b.E;
  }
} node[N];
int n, cnt[N];
int lowbit(int x) { return x & -x; }
void add(int i, int val) {
  while (i <= n) {
    c[i] += val;
    i += lowbit(i);
  }
}
int query(int i) {
  int s = 0;
  while (i > 0) {
    s += c[i];
    i -= lowbit(i);
  }
  return s;
}
int main() {
  while (scanf("%d", &n), n) {
    for (int i = 1; i <= n; ++i) {
      scanf("%d%d", &node[i].S, &node[i].E);
      node[i].index = i;
    }
    sort(node + 1, node + n + 1);
    memset(c, 0, sizeof(c));
    memset(cnt, 0, sizeof(cnt));
    cnt[node[1].index] - 0;
    add(node[1].S + 1, 1);
    for (int i = 2; i <= n; ++i) {
      if (node[i].E == node[i - 1].E && node[i].S == node[i - 1].S)
        cnt[node[i].index] = cnt[node[i - 1].index];
      else
        cnt[node[i].index] = query(node[i].S + 1);
      add(node[i].S + 1, 1);
    }
    for (int i = 1; i <= n; ++i) printf("%d ", cnt[i]);
    printf("\n");
  }
}

```

### 计数问题

```cpp
#include <bits/stdc++.h>
using namespace std;
int n, m;
const int N = 301;
int a[N][N];
int c[N][N][101];
inline void add(int x, int y, int k, int color) {
  for (int i = x; i <= n; i += i & -i)
    for (int j = y; j <= m; j += j & -j) c[i][j][color] += k;
}
inline int query(int x, int y, int color) {
  int ret = 0;
  for (int i = x; i; i -= i & -i)
    for (int j = y; j; j -= j & -j) ret += c[i][j][color];
  return ret;
}
int main() {
  scanf("%d%d", &n, &m);
  int q, x1, x2, y1, y2, color;
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= m; ++j) {
      scanf("%d", &color);
      a[i][j] = color;
      add(i, j, 1, color);
    }
  for (scanf("%d", &q); q--;) {
    int op;
    scanf("%d", &op);
    if (op == 1) {
      scanf("%d%d%d", &x1, &y1, &color);
      add(x1, y1, -1, a[x1][y1]);
      a[x1][y1] = color;
      add(x1, y1, 1, color);
    } else {
      scanf("%d%d%d%d%d", &x1, &x2, &y1, &y2, &color);
      printf("%d\n", query(x2, y2, color) - query(x1 - 1, y2, color) -
                         query(x2, y1 - 1, color) +
                         query(x1 - 1, y1 - 1, color));
    }
  }
}
```
