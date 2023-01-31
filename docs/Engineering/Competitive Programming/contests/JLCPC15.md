## K - Bracket Sequence

问由$K$种括号组成长度为$N$的不同的括号序列一共有多少个。

如果只有一种那么答案是卡特兰数$\binom{2N}{N} - \binom{2N}{N-1}$

如果是 K 种，那么其中任何一对括号可以被替换掉，即$\binom{2N}{N} - \binom{2N}{N-1}\times K^{N}$

```cpp
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <complex>
#include <cstdio>
#include <bitset>
#include <queue>
#include <stack>
#include <cmath>
#include <ctime>
#include <set>
#include <map>
using namespace std;
typedef long long ll;
typedef complex<double> CD;
#define MX maxn - 5
#define LS p<<1
#define RS p<<1|1
#define MP(a,b) make_pair((a),(b))
#define rep_(i,a,b) for(int i=(a); i<(b); ++i)
#define for_(i,a,b) for(int i=(a); i<=(b); ++i)
#define dwn_(i,a,b) for(int i=(a); i>=(b); --i)
inline void chkmax(int &x,int y) {if(x<y) x=y;}
inline void chkmin(int &x,int y) {if(x>y) x=y;}

const int maxn = 200005, inf = 0x3f3f3f3f ;
const ll Inf = 0x3f3f3f3f3f3f3f3fll;
const double eps = 1e-8, PI = acos(-1), e = 2.71828182845904523536, eu = 0.57721566490153286;
const ll P = 1e9 + 7;
int N, M, T, K, RT, cnt ;
vector<int> v[maxn]; //

ll jc[maxn];

ll fpow(int x) {
    ll tmp = N; ll bs = x, res = 1;
    while(tmp) {
        if(tmp & 1) res = res * bs % P;
        bs = bs * bs % P;
        tmp >>=1;
    }
    return res;
}
ll inv(ll x) {
    ll tmp = P - 2; ll bs = x, res = 1;
    while(tmp) {
        if(tmp & 1) res = res * bs % P;
        bs = bs * bs % P;
        tmp >>=1;
    }
    return res;
}
int main()
{
    //ios::sync_with_stdio(false), cin.tie(0);
    jc[0] = 1; cin >> N; cin >> K;
    for_(i,1,N * 2) jc[i] = jc[i - 1] * i % P;
    ll res1 = jc[2 * N] * inv(jc[N]) % P * inv(jc[N]) % P, res2 = jc[2 * N] * inv(jc[N + 1]) % P * inv(jc[N - 1]) % P;
    // printf("%lld\n", (res1 - res2 + P ) % P);
    printf("%lld", (res1 - res2 + P ) % P * fpow(K) %P);

    return 0;
}

//#include<bits/stdc++.h>
//按 Ctrl + Alt + N 编译运行
```

## L

只需要判断每个字符和串首是否相等就行了

```cpp
#include <bits/stdc++.h>
using namespace std;

int main(){
    cin.tie(0);
    ios::sync_with_stdio(false);
    int t;cin>>t;
    while(t--){
        string s;
        cin>>s;
        int sz=s.size();
        char c=s[0];
        int flg=1;
        for(int i=1;i<sz;i++){
            if(s[i]!=c){
                cout<<2*sz-i<<'\n';flg=0;break;
            }
        }
        if(flg)cout<<sz-1<<'\n';
    }
}
```

## G - Matrix Repair

一个$01$矩阵中出现数据丢失（以$-1$表示），给出原矩阵各行、各列的异或值，尝试唯一复原原矩阵。

问题实质是一个模线性方程，对于一个缺失值，只有该行和该列只有它一个缺失值才能被求解。

整个问题可以看成一个无向二分图，将行和列看作点，行$r$列$c$之间有一条边当且仅当$m[r][c] = -1$。

拓扑排序的步骤如下：把所有有一个缺失值的行、列的序号入队，之后逐个出队。如果该行/列已被修复（因为在修复行的时候也会修复列），`continue`，否则修复之，并将其所有所连节点的缺失值数减一。

```cpp
#include <bits/stdc++.h>
using namespace std;
int mp[1002][1002];
unordered_set<int>r[1002],c[1002];
int row[1002],col[1002];
int cntr[1002],cntc[1002];
queue<int>qr,qc;
int main(){
    cin.tie(nullptr);
    ios::sync_with_stdio(false);
    int n;cin>>n;
    for(int i=1;i<=n;i++){
        for(int j=1;j<=n;j++){
            cin>>mp[i][j];
            if(mp[i][j]==-1) {
                r[i].insert(j),c[j].insert(i);
                cntr[i]++,cntc[j]++;
            }
            else row[i]^=mp[i][j],col[j]^=mp[i][j];
        }
    }
    for(int i=1,x;i<=n;i++){
        cin>>x;row[i]^=x;
        if(cntr[i]==1&&cntc[*r[i].begin()]==1){
            mp[i][*r[i].begin()]=row[i];cntr[i]--;cntc[*r[i].begin()]--;continue;
        }
        if(cntr[i]==1){
            qr.push(i);
        }
    }
    for(int i=1,x;i<=n;i++){
        cin>>x;col[i]^=x;
        if(cntc[i]==1){
            qc.push(i);
        }
    }
    while(!qr.empty()||!qc.empty()){
        while(!qr.empty()){
            if(cntr[qr.front()]==0){qr.pop();break;}
            int r1=qr.front();
            int c1=*r[r1].begin();
            cntr[r1]--;cntc[c1]--;
            mp[r1][c1]=row[r1],col[c1]^=mp[r1][c1];
            r[r1].erase(c1);c[c1].erase(r1);
            if(c[c1].size()==1)qc.push(c1);
        }
        while(!qc.empty()){
            if(cntc[qc.front()]==0){qc.pop();break;}
            int c1=qc.front();
            int r1=*c[c1].begin();
            cntc[c1]--;cntr[r1]--;
            mp[r1][c1]=col[c1];row[r1]^=mp[r1][c1];
            c[c1].erase(r1);r[r1].erase(c1);
            if(r[r1].size()==1)qr.push(r1);
        }
    }
    for(int i=1;i<=n;i++){
        if(cntc[i]==0&&cntr[i]==0)continue;
        cout<<"-1"<<endl;return 0;
    }
    for(int i=1;i<=n;i++){
        for(int j=1;j<=n;j++){
            cout<<mp[i][j]<<' ';
        }
        cout<<'\n';
    }
    return 0;
}
```

## C

题目给出$a, x_1, b, x_i$，要求$x_i$是否满足$x_{i+1} \equiv a \times x_i + b\pmod{p}$。

化成一个等比数列

$$
x_{i+1} + \frac{b}{a-1} = a\times(x_i + \frac{b}{a-1}) \pmod{p}
$$

根据等比数列递推公式可得

$$
x_{i+1} + \frac{b}{a-1} = a^{n-1}\times(x_1 + \frac{b}{a-1}) \pmod{p}
$$

只有$a^{n-1}$是未知的，移项

$$
a^{n-1} \equiv (x_n + b \times \mathrm{inv}(a-1) \times \mathrm{inv}(x_1 + b\times \mathrm{inv}(a-1))) \pmod{p}
$$

然后就是一个[BSGS](https://oi-wiki.org/math/number-theory/bsgs/)。

参考：[SDOI 2013 随机数生成器](https://www.luogu.com.cn/problem/P3306)给出$a, x_1, b, x_i$，要求得到$x_{i+1} \equiv a \times x_i + b\pmod{p}$的最小$i$。

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
ll a,b,m,x,x0;
ll poww(ll a,ll b){
    ll ans=1;
    while(b){
        if(b&1)ans=ans*a%m;
        a=a*a%m,b>>=1;
    }
    return ans;}
unordered_map<ll,ll>pp;
bool bsgs(ll bas,ll k){
    ll mm=ceil(sqrt(m));
    ll tmp=k%m;
    for(int i=0;i<mm;i++){
        pp[tmp]=i;
        tmp=tmp*bas%m;
    }
    ll dy=poww(bas,mm);
    tmp=1;
    for(int i=0;i<=mm;i++){
        if(pp.find(tmp)!=pp.end())return 1;
        tmp=tmp*dy%m;
    }
    return 0;
}
int main(){
    cin.tie(nullptr);
    ios::sync_with_stdio(false);
    cin>>a>>b>>m>>x0>>x;
    if(a==0){
        if(b==x||x0==x){cout<<"YES"<<endl;return 0;}
        else{cout<<"NO"<<endl;return 0;}
    }
    if(a==1){
        if(b!=0||b==0&&x0==x){cout<<"YES"<<endl;return 0;}
        else {cout<<"NO"<<endl;return 0;}
    }
    if(b+x0*(a-1)==0){
        if(x==x0){cout<<"YES"<<endl;return 0;}
        else {cout<<"NO"<<endl;return 0;}
    }
    ll k=(m+x-x0)%m*poww((b*poww(a-1,m-2)+x0)%m,m-2)%m;
    k++;
    k%=m;
    bool ans=bsgs(a,k);
    if(ans==0)cout<<"NO"<<endl;
    else cout<<"YES"<<endl;
    return 0;
}
```

## H - Visit the Park

题意：给定一个带 0 ∼ 9 边权的图和一条路径，询问沿路径经过边的边权形成数字的期望值。

计算每一条边边权的期望值，之后沿路径统计答案即可。

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N =300010;
typedef long long ll;
const int mod=998244853;
map<pair<int,int>,ll>dig;
map<pair<int,int>,ll>nm;
ll wei[N],a[N];
ll ny(ll b){
    ll x=mod-2,ans=1;
    while(x){
        if(x&1)ans=ans*b%mod;
        b=b*b%mod;
        x>>=1;
    }
    return ans;
}

int main(){
    cin.tie(0);
    ios::sync_with_stdio(false);
    int n,m,k;
    cin>>n>>m>>k;
    for(int i=1,u,v,w;i<=m;i++){
        cin>>u>>v>>w;
        dig[make_pair(u,v)]+=w;
        nm[make_pair(u,v)]++;
        dig[make_pair(v,u)]+=w;
        nm[make_pair(v,u)]++;
    }
    ll ans=0;
    wei[k]=1;
    for(int i=k-1;i>=2;i--){
        wei[i]=wei[i+1]*10%mod;
    }
    for(int i=1;i<=k;i++)cin>>a[i];
    for(int i=2;i<=k;i++){
        if(nm[make_pair(a[i],a[i-1])]==0){
            cout<<"Stupid Msacywy!\n";return 0;
        }
        else{
            ll dg=dig[make_pair(a[i],a[i-1])],nmb=nm[make_pair(a[i],a[i-1])];
            ans+=wei[i]*ny(nmb)%mod*dg%mod;
        }
    }
    cout<<ans%mod<<endl;
    return 0;
}
```
