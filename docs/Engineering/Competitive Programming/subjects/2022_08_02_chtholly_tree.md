# 颜色段均摊（珂朵莉树）

<https://blog.csdn.net/CC_dsm/article/details/98166835>

> 珂朵莉树并不是一种新的树，甚至不是一种数据结构，指代一种特定的基于数据随机的算法。
>
> > 实际上这种平凡的东西还是别起名字比较好，或者就叫“基于数据随机的**颜色段均摊**”之类的应该是更适合的。

在具有*区间赋值*，*区间统计*操作，以及最好保证**数据随机**的情况下在*时空复杂度*上把线段树吊起来打（详情见后）。

```cpp
template<typename type>
struct Node{
	unsigned int l;
	unsigned int r;
	mutable type data;
	Node(unsigned int a, unsigned int b = 0, type c = 0);
	bool operator <(const Node &a) const
	{
		return l < a.l;
	}
};

Node::Node(unsigned int a, unsigned int b, type c){
    l = a;
    r = b;
    data = c;
}
```

解释一下上面的代码。

- 珂朵莉树的每一个节点代表着一个闭区间，那么 `Node` 结构体里理应有这个区间的左右边界（即 `l` 和 `r`）。
- `type` 和 `data` 是当前区间统一的类型与数值，就是说闭区间[l,r]内每个点的类型都是 `type`，值都是 `data`。（当然，我们只考虑离散的点）
- `data` 需要 `mutable` 修饰，这样我们可以在 `set` 中利用迭代器修改它。
- 对于结构体，我们自然需要构造函数，无需多讲。
- 我们使用 `set` 来存储 `Node`，所以我们需要重载小于号，使其按照左端点排序。

### 构造珂朵莉树

```cpp
#include <set>
set<Node> s;
```

就这么简单，你得到了一个没有初始化的珂朵莉树。
**一般来说**，我们通过给定数据，向其中不断插入区间长度为 1 的区间来完成初始化。

比如形如这样的话：“第二行包括 n 个数，表示序列的初始状态”（摘自 SCOI2010 序列操作）。

我们就可以这样初始化：

```cpp
for (int i = 0; i < n; ++i) {
 	int temp = 0;
 	cin >> temp;
 	s.insert(Node(i, i, temp));
 }
 s.insert(Node(n, n, 0));
```

你的序列下标从 0 或者 1 开始是无所谓的。

这里有一个蜜汁细节，就是在把所有给定数据插入完成之后，需要在末尾多插入一个节点。我也不知道这究竟有啥用，根据自己测试貌似做不做这一步并没有什么区别，反正是玄学，信就完事了。

# 核心操作

## 分裂：split

既然我们要进行区间操作，那就得把这个区间**拿出来**（就是这么暴力的思想） 。
`split(pos)` 操作将包含位置 $pos$ 的区间 $[l,r]$ 分裂成 $[l,pos-1]$ 和 $[pos,r]$，并返回后者的迭代器。

```cpp
auto split(unsigned int pos)
{
 auto it = s.lower_bound(Node(pos));
 if (it != s.end() && it->l == pos)
  return it;
 --it;
 unsigned int l = it->l, r = it->r;
 auto data = it->data;
 s.erase(it);
 s.insert(Node(l, pos - 1, data));
 return s.insert(Node(pos, r, data)).first;
}
```

我们先利用 lower_bound()函数在 set 中查到左端点位置大于等于 pos 的节点。

如果这个节点的左端点位置正是 pos，那么我们无需分裂，直接返回。

如果它的左端点位置不是 pos，那么必然大于 pos，则包含位置 pos 的节点是上一个节点，it-=1。

接下来的事情就好办了，暴力分裂再插入即可。不要忘了返回值。

此时，如果我们想使用区间[l,r]中的数据，只需要这么写：

```cpp
auto it2 = split(r + 1), it1 = split(l);
 for (; it1 != it2; ++it1) {
   //利用迭代器it1搞些事情
 }
```

这里有一个细节必须注意，必须**先声明 it2 再声明 it1**，否则根据 `split` 中的 `erase` 操作，迭代器 `it1` **可能会**失效。（因为 `it1` 所属的节点可能被删除了）

## 区间赋值：assign

珂朵莉树最重要的操作，也是不让它退化为暴力算法的玄学 保障。
**既然一个区间内所有的值全都一样了，那么在珂朵莉树中这个区间就可以只用一个节点来表示**。这就是珂朵莉树的核心，光速降低节点数量的神器。

```cpp
void assign(unsigned int l, unsigned int r, type val)
{
 auto it2 = split(r + 1), it1 = split(l);
 s.erase(it1, it2);
 s.insert(Node(l, r, val));
 return;
}
```

可见，这个区间里所有的节点全部被删除，使用一个新的节点来代替。

根据~~我并不会的~~证明，`assign` 的区间长度在随机数据下的期望为 N/3，十分恐怖。

而且这个 `assign` 在赋值之余还可以顺便做做区间统计啥的，根据情况而定

至此，珂朵莉树的核心操作介绍完毕。

附加的工作？

很多时候，一道题不可能只用两个函数就轻松搞定，需要额外的暴力函数与算法，是的就是暴力。

由于暴力算法大家肯定会，又怕大家不好理解，所以在这里贴一下 CF896C 的代码。

这道题虽说是起源，但是还是比较有难度的.

```cpp
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <set>
#include <vector>

using namespace std;

#define S_IT set<Node>::iterator
typedef long long ll;

struct Node
{
	int l, r;
	mutable ll val;
	bool operator <(const Node &a) const
	{
		return l < a.l;
	}
	Node(int a, int b, ll v);
	Node(int a);
};

S_IT split(int pos);
void add(int l, int r, int v);
ll kth(int l, int r, int k);
ll qpow(ll a, int b, ll y);
ll query(int l, int r, int x, ll y);
void assign(int l, int r, int v);
int rnd(void);

set<Node> s;
ll seed;
int n, m, vmax;

int main(void)
{
	ios::sync_with_stdio(false);
	cin >> n >> m >> seed >> vmax;
	for (int i = 1; i <= n; ++i)
	{
		static int temp = 0;
		temp = rnd() % vmax + 1;
		s.insert(Node(i, i, (ll)temp));
	}
	s.insert(Node(n + 1, n + 1, 0));
	for (int i = 1; i <= m; ++i)
	{
		static int l = 0, r = 0, x = 0, y = 0, op = 0;
		op = rnd() % 4 + 1;
		l = rnd() % n + 1, r = rnd() % n + 1;
		if (l > r)
		{
			swap(l, r);
		}
		if (op == 3)
		{
			x = rnd() % (r - l + 1) + 1;
		}
		else
		{
			x = rnd() % vmax + 1;
		}
		if (op == 4)
		{
			y = rnd() % vmax + 1;
		}

		if (op == 1)
		{
			add(l, r, (ll)x);
		}
		else if (op == 2)
		{
			assign(l, r, (ll)x);
		}
		else if (op == 3)
		{
			cout << kth(l, r, x) << endl;
		}
		else if (op == 4)
		{
			cout << query(l, r, x, (ll)y) << endl;
		}
	}
	//system("pause");
	return 0;
}

Node::Node(int a, int b, ll v)
{
	l = a;
	r = b;
	val = v;
}

Node::Node(int a)
{
	l = a;
}

S_IT split(int pos)
{
	S_IT it = s.lower_bound(Node(pos));
	if (it != s.end() && it->l == pos)
	{
		return it;
	}
	--it;
	int l = it->l, r = it->r;
	ll val = it->val;
	s.erase(it);
	s.insert(Node(l, pos - 1, val));
	return s.insert(Node(pos, r, val)).first;
}

void add(int l, int r, int v)
{
	S_IT it2 = split(r + 1), it1 = split(l);
	for (S_IT it=it1; it != it2; ++it)
	{
		it->val += v;
	}
}

ll kth(int l, int r, int k)
{
	S_IT it2 = split(r + 1), it1 = split(l);
	vector<pair<ll, int> >arr;
	arr.clear();
	for (S_IT it = it1; it != it2; ++it)
	{
		arr.push_back(pair<ll, int>(it->val, it->r - it->l + 1));
	}
	sort(arr.begin(), arr.end());
	for (unsigned int i = 0; i < arr.size(); ++i)
	{
		k -= arr[i].second;
		if (k <= 0)
		{
			return arr[i].first;
		}
	}
}

ll qpow(ll a, int x, ll y)
{
	ll b = 1LL;
	a %= y;
	while (x)
	{
		if (x & 1)
		{
			b = (b*a) % y;
		}
		a = (a*a) % y;
		x >>= 1;
	}
	return b;
}

ll query(int l, int r, int x, ll y)
{
	S_IT it2 = split(r + 1), it1 = split(l);
	ll res = 0;
	for (S_IT it = it1; it != it2; ++it)
	{
		res = (res + (it->r - it->l + 1)*qpow(it->val, x, y)) % y;
	}
	return res;
}

void assign(int l, int r, int v)
{
	S_IT it2 = split(r + 1), it1 = split(l);
	s.erase(it1, it2);
	s.insert(Node(l, r, v));
}

int rnd(void)
{
	int ret = (int)seed;
	seed = (seed * 7 + 13) % 1000000007;
	return ret;
}
```
