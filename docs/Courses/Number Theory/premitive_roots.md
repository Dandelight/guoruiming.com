# 原根

## 整数的次数

设 $m>0$，$(a, m) = 1$，$l$ 是使

$$
a^l \equiv 1 \pmod m
$$

成立的最小正整数，则 $l$ 是 $a$ 对模数 $m$ 的原根。

**定理 2**：设 $a$ 对模数 $m$ 的次数为 $l$，则

$$
1, a, a^2, \ldots a^{l-1}
$$

对模数 $m$ 两两不同余。

**定理 3**：设 $a$ 对模数 $m$ 的次数为 $l$，$\lambda > 0$，$a^\lambda$ 对模数 $m$ 的次数为 $l_1$，则 $l_1 = \dfrac{l}{(\lambda, l)}$。

**推论**：设 $a$ 为对模数 $m$ 的次数为 $l$，则 $\varphi(l)$ 个数

$$
a^\lambda,\ (\lambda, l)=1,\ 0 < \lambda \le l
$$

对模数 $m$ 的次数均为 $l$。

**定理 4**：设 $p$ 是一个素数，如果存在整数 $a$，它对模数 $p$ 的次数是 $l$，则恰有 $\varphi(l)$ 个对模数 $p$ 两两不同余的整数，它们对模数 $p$ 的次数都为 $l$。

**定理 5**：设 $l \mid p -1$，则次数为 $l$ 的，模数 $p$ 互不同余的整数的个数是 $\varphi(l)$ 个。

## 原根

**定义**：设整数 $m>0$，$(g, m) = 1$，如果整数 $g$ 对 $m$ 的次数是 $\varphi(m)$，则 $g$ 叫做 $m$ 的一个原根。

**定理 1**：设 $(g, m) = 1$，$m>0$，则 $g$ 是 $m$ 的一个原根的充分必要条件是

$$
g, g^2, \ldots, g^{\varphi(m)}
$$

组成模数 $m$ 的一组缩系。

**定理 2**：设 $m > 1$，若 $m$ 有原根，则 $m$ 必为下列诸数之一：

$$
2, 4, p^l, p^{2l}
$$

这里 $l \ge 1$，$p$ 为奇素数。

**定理 3**：$m = 2, 4, p^l, p^{2l}$（$l\ge 1$， $p$ 为奇素数）时，$m$ 有原根。

**引理**：设 $g$ 是奇素数 $p$ 的一个原根，满足

$$
g^{p-1} \not\equiv 1 \pmod{p^2}
$$

则对于每一个 $a \ge 2$，有

$$
g^{\varphi(p^{\alpha-1})} \not\equiv 1 \pmod{p^\alpha}
$$

**定理 4**：设 $m$ 有一个原根 $g$，则 $m$ 恰有 $\varphi(\varphi(m))$ 个对模数 $m$ 互不同余的原根，它们是由集

$$
S = \{ g^t \mid 1 \le t \le \varphi(m),\ (t, \varphi(m)) = 1\}
$$

中的数给出。

## 次数的计算

**定理 1**：如果 $m = p_1^{l_1}\ldots p_k^{l_k}$ 是 $m$ 的标准分解式，整数 $a$ 对模数 $m$ 的次数等于整数 $a$ 对模数 $p_i^{l_i}\ (i=1, \ldots, k)$ 的诸次数的最小公倍数。

**定理 2**：设 $p$ 是一个素数，$a$ 对模数 $p^j$ 的次数是 $f_j$，又设 $p_i \mid\mid a^{f_2} - 1$，有

$$
f_j = \begin{cases}
f_2 & \text{if}\quad 2\le j \le i\\
p^{j-i} f_2 & \mathrm{if} \quad j>i
\end{cases}
$$

## 原根的计算

**定理 1**：设 $m>2$ ，$\varphi(m)$ 的所有不同的素因子是 $q_1, q_2, \ldots, q_s$，$(g, m) = 1$，则 $g$ 是 $m$ 的一个原根的充分必要条件是

$$
g^{\frac{\varphi(m)}{q_i}} \neq 1 \pmod{m} \quad(i=1, 2, \ldots, s)
$$

**定理 2**：设 $a$ 对模数奇素数 $p$ 的次数是 $d$，$d<p-1$，则

$$
a^\lambda, \quad \lambda=1,2,\ldots,d
$$

都不是 $p$ 的原根。

要求出原根，先列出各数

$$
1,2,\ldots p-1
$$

取 $a=2$，计算 $2$ 对 $p$ 的次数 $d$，如果 $d = p-1$，$2$ 就是 $p$ 的原根。如果 $d < p-1$，在上式中除去下列各数：

$$
<2>_p, <2^2>_p, \ldots, <2^d>_p
$$

在上式中再取一数，重复上述方法，直到剩下 $\varphi(p-1)$ 个数，这些数都是 $p$ 的原根。

```cpp
int gcd(int a, int b) { return a ? gcd(b % a, a) : b; }

int powmod(int a, int b, int p) {
  int res = 1;
  while (b > 0) {
    if (b & 1) res = res * a % p;
    a = a * a % p, b >>= 1;
  }
  return res;
}

// Finds the primitive root modulo p
int generator(int p) {
  vector<int> fact;
  int phi = p - 1, n = phi;
  for (int i = 2; i * i <= n; ++i) {
    if (n % i == 0) {
      fact.push_back(i);
      while (n % i == 0) n /= i;
    }
  }
  if (n > 1) fact.push_back(n);
  for (int res = 2; res <= p; ++res) {
    bool ok = true;
    for (int factor : fact) {
      if (powmod(res, phi / factor, p) == 1) {
        ok = false;
        break;
      }
    }
    if (ok) return res;
  }
  return -1;
}

// This program finds all numbers x such that x^k=a (mod n)
int main() {
  int n, k, a;
  scanf("%d %d %d", &n, &k, &a);
  if (a == 0) return puts("1\n0"), 0;
  int g = generator(n);
  // Baby-step giant-step discrete logarithm algorithm
  int sq = (int)sqrt(n + .0) + 1;
  vector<pair<int, int>> dec(sq);
  for (int i = 1; i <= sq; ++i)
    dec[i - 1] = {powmod(g, i * sq * k % (n - 1), n), i};
  sort(dec.begin(), dec.end());
  int any_ans = -1;
  for (int i = 0; i < sq; ++i) {
    int my = powmod(g, i * k % (n - 1), n) * a % n;
    auto it = lower_bound(dec.begin(), dec.end(), make_pair(my, 0));
    if (it != dec.end() && it->first == my) {
      any_ans = it->second * sq - i;
      break;
    }
  }
  if (any_ans == -1) return puts("0"), 0;
  // Print all possible answers
  int delta = (n - 1) / gcd(k, n - 1);
  vector<int> ans;
  for (int cur = any_ans % delta; cur < n - 1; cur += delta)
    ans.push_back(powmod(g, cur, n));
  sort(ans.begin(), ans.end());
  printf("%d\n", ans.size());
  for (int answer : ans) printf("%d ", answer);
}s
```

## 原根的一个性质

**定理 1**：设 $p$ 是一个奇素数，$\mathrm Q(p)$ 表示 $p$ 的 互不同余的 $\varphi(p-1)$ 个原根的和，我们有

$$
\mathrm{Q}(p) \equiv \mu(p-1) \pmod{p}
$$

其中$\mu(\cdot)$表示莫比乌斯函数。

**定理 2**：设 $p$ 为一个奇素数，对模数 $p$ 的次数为 $d$ 的 $\varphi(d)$ 个互不同余的数的 $r$ 次幂的和为 $S$，则

$$
S \equiv \dfrac{\varphi(d)}{\varphi(d_1)} \mu(d_1) \pmod(p)
$$

这里 $d_1 = \dfrac{d}{(r, d)}$。
