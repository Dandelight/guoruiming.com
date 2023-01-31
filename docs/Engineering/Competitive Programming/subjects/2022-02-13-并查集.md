# 什么集来着？

| 题目                                   | 类型       | 链接                                                                              |
| -------------------------------------- | ---------- | --------------------------------------------------------------------------------- |
| UVa1160 X-Plosives                     | 模板题     | https://www.luogu.com.cn/problem/UVA1160                                          |
| USACO2016 DecG P1. Moocast             | MST        | https://www.luogu.com.cn/problem/P2847                                            |
| USACO2016 OpenG P2. Closing the Farm   | 离线       | https://www.luogu.com.cn/problem/P6121                                            |
| POJ1962 Corporative Network            | 加权并查集 | http://bailian.openjudge.cn/practice/1962?lang=en_US                              |
| USACO2020 Jan S P3. Wormhole Sort      | 连通性     | [https://www.luogu.com.cn/problem/P6004 ](https://www.luogu.com.cn/problem/P6004) |
| USACO2013 Feb S P2. Tractor            | MST        | [https://www.luogu.com.cn/problem/P3073 ](https://www.luogu.com.cn/problem/P3073) |
| USACO2018 Jan G P1. MooTube            | 离线算法   | https://www.luogu.com.cn/problem/P4185                                            |
| USACO 2014 Jan G P3. Ski Course Rating | 连通性     | [https://www.luogu.com.cn/problem/P3101 ](https://www.luogu.com.cn/problem/P3101) |
| UVa10158 - War                         | 二分图     | https://www.luogu.com.cn/problem/UVA10158                                         |
| USACO2020 Open G P2. Favorite Colors   | 图的合并   | https://www.luogu.com.cn/problem/P6279                                            |
| Baltic OI 2016 Park                    | 几何思维   | https://www.luogu.com.cn/problem/P4675                                            |

## UVa1160 X-Plosives

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 1e5 + 10;
int par[N];
int find(int x) { return par[x] = par[x] == x ? x : find(par[x]); }
int main() {
  while (true) {
    for (int i = 1; i <= (int)1e5; ++i) par[i] = i;
    vector<pair<int, int>> ex;
    while (true) {
      int n, m;
      if (cin >> n) {
        if (n != -1)
          cin >> m;
        else
          break;
        ex.push_back({n, m});
      } else
        return 0;
    }
    int cnt = 0;
    for (auto &it : ex) {
      auto u = it.first, v = it.second;
      u = find(u);
      v = find(v);
      if (u == v)
        cnt++;
      else
        par[u] = v;
    }
    cout << cnt << endl;
  }
}
```

## Moocast

```cpp
#include <bits/stdc++.h>
typedef long long ll;
using namespace std;
const int N = 1010;
int a[N], b[N];
int par[25010];
int find(int x) { return par[x] = x == par[x] ? x : find(par[x]); }
struct ed {
  int u, v;
  ll w;
  bool operator<(const ed &b) const { return w < b.w; }
};
ll dist(int i, int j) {
  ll dx = a[i] - a[j];
  ll dy = b[i] - b[j];
  return dx * dx + dy * dy;
}
vector<ed> E;
int n;
int main() {
  for (int i = 0; i <= 25000; ++i) par[i] = i;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> a[i] >> b[i];
  }
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      E.push_back({i, j, dist(i, j)});
    }
  }
  sort(E.begin(), E.end());

  int sel = 0;
  for (auto [u, v, w] : E) {
    u = find(u);
    v = find(v);
    if (u == v) continue;
    par[u] = v;
    sel++;
    if (sel == n - 1) {
      cout << w << endl;
      return 0;
    }
  }
}

```

## Closing the farm

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef pair<int, int> pi;
const int N = 2e5 + 10;
vector<int> adj[N];
vector<int> close;
int par[N];
bool ans[N];
bool check[N];
int find(int x) { return par[x] = x == par[x] ? x : find(par[x]); }
int main() {
  int n, m;
  cin >> n >> m;
  for (int i = 1; i <= n; ++i) par[i] = i;
  for (int i = 0; i < m; ++i) {
    int u, v;
    cin >> u >> v;
    adj[u].push_back(v);
    adj[v].push_back(u);
  }
  for (int i = 0; i < n; ++i) {
    int c;
    cin >> c;
    close.push_back(c);
  }
  // 倒序遍历close，逐个“打开”
  int cc = 0;  // Connected Composnets
  for (int i = n - 1; i >= 0; --i) {
    cc++;
    int u = close[i];
    check[u] = true;
    for (auto v : adj[u]) {
      if (check[v]) {
        int fu = find(u), fv = find(v);
        if (fu != fv) {
          cc--;
          par[fu] = fv;
        }
      }
    }
    if (cc == 1) ans[i] = true;
  }
  for (int i = 0; i < n; ++i) {
    cout << (ans[i] ? "YES\n" : "NO\n");
  }
  return 0;
}
```

## Corporative Network

```cpp
#include <bits/stdc++.h>
using namespace std;
int t, n;
char str[5];
const int N = 20005;
int par[N], dist[N];
int find(int x) {
  if (x == par[x]) return x;
  int fx = find(par[x]);  // 这行不能写在更新dist之后
  dist[x] = dist[x] + dist[par[x]];
  return par[x] = fx;
}
void unite(int x, int y) {
  int fx = find(x), fy = find(y);
  if (fx == fy) return;
  par[fx] = y;
  dist[fx] = dist[x];
  dist[x] = abs(x - y) % 1000;
}
int main() {
  int i, j, k, x, y;
  cin >> t;
  while (t--) {
    cin >> n;
    for (int i = 0; i <= n; ++i) {
      par[i] = i;
      dist[i] = 0;
    }
    while (cin >> str) {
      if (*str == 'O')
        break;
      else if (*str == 'E') {
        cin >> x;
        find(x);
        cout << dist[x] << '\n';
      } else {
        cin >> x >> y;
        unite(x, y);
      }
    }
  }
}
```

## Wormhole sort

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 100020;
int n, m;
int par[N];
int arr[N];  // 这题不是最小生成树，而是将需要交换的奶牛都连起来
struct edge {
  int a, b, w;
  bool operator<(const edge& x) const { return w > x.w; }
} h[N];
int find(int x) { return par[x] = x == par[x] ? x : find(par[x]); }
int main() {
  cin >> n >> m;
  for (int i = 1; i <= n; ++i) par[i] = i;
  for (int i = 1; i <= n; ++i) cin >> arr[i];
  for (int i = 1; i <= m; ++i) cin >> h[i].a >> h[i].b >> h[i].w;
  int ans = -1, cnt = 0;
  sort(h + 1, h + m + 1);
  for (int i = 1; i <= m; ++i) {
    // 判断是否连通
    while (find(cnt) == find(arr[cnt])) {
      cnt++;
      if (cnt >= n) {
        goto End;
      }
    }
    auto [a, b, w] = h[i];
    int fa = find(a), fb = find(b);
    if (fa != fb) {
      par[fa] = fb;
      ans = w;
    }
  }
End:
  cout << ans << endl;
  return 0;
}
```

## Tractor

```cpp
// 嗯，体力活没错了
#include <bits/stdc++.h>
using namespace std;
const int N = 510;
int n;
int g[N * N];
int par[N * N];
int siz[N * N];
struct edge {
  int u, v, w;
  bool operator<(const edge &x) const { return w < x.w; }
};
vector<edge> e;
void add(int i, int j, int ti, int tj) {
  int u = n * i + j;
  int v = n * ti + tj;
  int fall = abs(g[u] - g[v]);
  e.push_back({u, v, fall});
}
int find(int x) { return par[x] = x == par[x] ? x : find(par[x]); }
int main() {
  cin >> n;
  for (int i = 0; i <= n * n; ++i) {
    par[i] = i;
    siz[i] = 1;
  }
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) cin >> g[n * i + j];

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      int ti = i + 1, tj = j;
      if (ti < n) add(i, j, ti, tj);
      ti--, tj++;
      if (tj < n) add(i, j, ti, tj);
    }
  }
  sort(e.begin(), e.end());
  int sel = 0, ans = -1;
  for (int i = 0; i < e.size(); ++i) {
    auto [u, v, w] = e[i];
    int fu = find(u), fv = find(v);
    if (fu == fv) continue;
    par[fu] = fv;
    siz[fv] += siz[fu];
    if (siz[fv] >= (n * n + 1) / 2) {
      ans = w;
      break;
    }
  }
  cout << ans << endl;
}
```

## Ski Course Rating

```cpp
#include<bits/stdc++.h>
using namespace std;
long long dx[3]={0, 1, 0}, dy[3]={0, 0, 1};
long long n, m, t;
long long cnt, tot;
long long ans;
struct node{
    long long x, y, dis;
}a[250005];
long long f[505][505];
long long father[250005],size[250005];
long long num[505][505],v[250005];
long long find(long long x)
{
  if(x!=father[x])
    father[x]=find(father[x]);
  return father[x];
}
bool cmp(node x,node y)
{
  return x.dis<y.dis;
}
int  main()
{
    for(int i=1;i<=250004;i++)
      size[i]=1,father[i]=i;
    scanf("%lld%lld%lld",&n,&m,&t);
    tot=0;
    for (int i=1;i<=n;i++)
    {
        for (int j=1;j<=m;j++)
        {
            scanf("%lld",&f[i][j]);
            tot+=1;
            num[i][j]=tot;
        }
    }
    for (int i=1;i<=n;i++)
    {
        for (int j=1;j<=m;j++)
        {
            int flag;
            scanf("%d",&flag);
            if(flag) v[num[i][j]]=1;
            for (int k=1;k<=2;k++)
            {
                int tx=i+dx[k],ty=j+dy[k];
                if (tx<n+1&&ty<m+1) a[++cnt]=(node){num[i][j],num[tx][ty],abs(f[i][j]-f[tx][ty])};
            }
        }
    }//建图
    sort(a+1,a+1+cnt,cmp);//排序
    for(int i=1;i<=cnt;i++)//不断加边
    {
        int x=a[i].x,y=a[i].y;
        int fx=find(x), fy=find(y);
        if(fx==fy)continue;
        if (size[fx]+size[fy]>=t)
        {
            if (size[fx]<t)ans+=a[i].dis*v[fx];
            if (size[fy]<t)ans+=a[i].dis*v[fy];
        }
        if (size[fx]>size[fy]) swap(fx,fy);
        father[fx]=fy;
        size[fy]+=size[fx],v[fy]+=v[fx];
    }
    printf("%lld",ans);//输出答案
    return 0;
}
```

## Wars

```cpp
#include <iostream>
#include <cstdlib>
#include <cstdio>

using namespace std;

//union
int sets[20001];

int Find( int x )
{
	if ( x != sets[x] )
		sets[x] = Find( sets[x] );
	return sets[x];
}
//union end

int main()
{
	int n,c,x,y;
	while ( scanf("%d",&n) != EOF ) {
		for ( int i = 0 ; i < 2*n ; ++ i )
			sets[i] = i;
		while ( scanf("%d%d%d",&c,&x,&y) && c ) {
			int a1 = Find( x ),a2 = Find( x+n );
			int b1 = Find( y ),b2 = Find( y+n );
			switch( c ) {
				case 1: if ( a1 == b2 ) printf("-1\n");
						else {
							sets[a1] = b1;
							sets[a2] = b2;
						}break;
				case 2: if ( a1 == b1 ) printf("-1\n");
						else {
							sets[a1] = b2;
							sets[a2] = b1;
						}break;
				case 3: if ( a1 == b1 ) printf("1\n");
						else printf("0\n");
						break;
				case 4: if ( a1 == b2 ) printf("1\n");
						else printf("0\n");
						break;
			}
		}
	}

	return 0;
}
```

## Favorate Colors G

```cpp

#include<bits/stdc++.h>
#define pb push_back
using namespace std;
const int maxn=200002;
inline int read()
{
    register int x=0;
    register char c=getchar();
    for(;!(c>='0'&&c<='9');c=getchar());
    for(;c>='0'&&c<='9';c=getchar())
        x=(x<<1)+(x<<3)+c-'0';
    return x;
}
int n,m;
vector<int>v[maxn];//存边
vector<int>son[maxn];
//son[i]:若i号奶牛是其所在节点的（那棵树的）根节点，
//那么存储所在节点的（那棵树的）所有节点
queue<int>q;
int fa[maxn];//fa[i]:i号奶牛所在节点的（那棵树的）根节点
int Vis[maxn],ans[maxn];
void hb(int x,int y)
{
	x=fa[x],y=fa[y];//对根节点进行操作
	if(son[x].size()<son[y].size())//启发式合并核心代码
		x^=y,y^=x,x^=y;
	for(register int i=0;i<v[y].size();i++)
		v[x].pb(v[y][i]);//合并
	for(register int i=0;i<son[y].size();i++)
		fa[son[y][i]]=x,son[x].pb(son[y][i]);//合并
	if(v[x].size()>1)
      //如果仰慕的奶牛超过一头，需要合并，入队列
		q.push(x);
}

int main()
{
	register int x,y,t,ru;
    n=read(),m=read();
    for(register int i=1;i<=m;i++)
    	x=read(),y=read(),v[x].pb(y);
   	for(register int i=1;i<=n;i++)
	{
		fa[i]=i;//一开始每个奶牛所在节点的根节点都是它自己
		if(v[i].size()>1)
     //如果有超过一头奶牛仰慕当前奶牛，那么需要合并，加入队列
   		q.push(i);
   		son[i].pb(i);
      //一开始每个奶牛都是所在节点的根节点节点，且节点中必定有这头奶牛（废话）
	}
   	while(!q.empty())
   	{
   		t=q.front(),q.pop();
	   	while(v[t].size()>1)
	   	{
	   		ru=v[t][1],x=v[t][0],v[t].erase(v[t].begin());
	   		if(fa[ru]!=fa[x])
         //如果不是同一节点中的奶牛再合并，防止计算重复
				hb(ru,x);
		}
   	}
   	register int col=0;
   	for(register int i=1;i<=n;i++)
   	{
   		if(Vis[fa[i]]==0)
   			Vis[fa[i]]=++col;
   		cout<<Vis[fa[i]]<<endl;
   	}
    return 0;
}
```

## Park

```cpp
#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
using namespace std;
const int MAXN=2015;
const int MAXM=100015;
const double eps=1e-4;
int n,m,id,last,k;
long long l,r;

struct tree
{
    long long x,y,d;

    inline double operator - (struct tree tmp)
    {
        return sqrt((x-tmp.x)*(x-tmp.x)+(y-tmp.y)*(y-tmp.y))-d-tmp.d;
    }
}t[MAXN];

struct man
{
    long long from,id,d;

    inline bool operator < (const man &tmp) const
    {
        return d<tmp.d;
    }
}w[MAXM];

struct cost
{
    long long a,b;double dis;

    inline bool operator < (const cost &tmp) const
    {
        return dis<tmp.dis;
    }
}h[MAXN*MAXN];
int f[MAXM];
bool ans[MAXM][5],map[5][5];

inline int find(int x)
{
    return f[x]==x?x:f[x]=find(f[x]);
}

inline void together(int x,int y)
{
    int r1,r2;
    r1=find(x);r2=find(y);
    if (r1==r2) return ;
    f[r1]=r2;
}

inline void turn_off(int x,int y)
{
    map[x][y]=map[y][x]=false;
}

int main()
{
    cin>>n>>m>>l>>r;
    for (int i=1;i<=n;i++) cin>>t[i].x>>t[i].y>>t[i].d;
    for (int i=1;i<=m;i++) {cin>>w[i].d>>w[i].from;w[i].d*=2;w[i].id=i;}
    for (int i=1;i<=n;i++)
    {
        h[++id]=(cost){i,n+1,(double)t[i].x-t[i].d};
        h[++id]=(cost){i,n+2,(double)t[i].y-t[i].d};
        h[++id]=(cost){i,n+3,(double)l-t[i].x-t[i].d};
        h[++id]=(cost){i,n+4,(double)r-t[i].y-t[i].d};
        for (int j=i+1;j<=n;j++) h[++id]={i,j,fabs(t[i]-t[j])};
    }
    last=1;
    sort(h+1,h+id+1); sort(w+1,w+m+1);
    for (int i=1;i<=4;i++)
        for (int j=1;j<=4;j++) map[i][j]=true;
    for (int i=1;i<=n+10;i++) f[i]=i;
    for (int i=1;i<=m;i++)
    {
        while (last<=id&&h[last].dis+eps<=w[i].d) {together(h[last].a,h[last].b);last++;}
        if (find(n+1)==find(n+3)) turn_off(1,3),turn_off(1,4),turn_off(2,3),turn_off(2,4);
        if (find(n+2)==find(n+4)) turn_off(1,2),turn_off(1,3),turn_off(2,4),turn_off(3,4);
        if (find(n+1)==find(n+2)) turn_off(1,2),turn_off(1,3),turn_off(1,4);
        if (find(n+2)==find(n+3)) turn_off(1,2),turn_off(2,4),turn_off(2,3);
        if (find(n+3)==find(n+4)) turn_off(3,1),turn_off(3,2),turn_off(3,4);
        if (find(n+4)==find(n+1)) turn_off(4,1),turn_off(4,2),turn_off(4,3);
        for (int j=1;j<=4;j++) ans[w[i].id][j]=map[w[i].from][j];
    }
    for (int i=1;i<=m;i++)
    {
        for (int j=1;j<=4;j++)
            if (ans[i][j]) putchar(j+'0');
        putchar('\n');
    }
    return 0;
}
```
