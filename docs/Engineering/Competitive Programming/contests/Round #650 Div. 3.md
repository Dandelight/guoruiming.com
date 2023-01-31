## E - Necklace Assembly

一共有$n$种珠子，每颗珠子的颜色由一个小写英文字符表示。一条项链由一颗或多颗柱子首尾相连组成。当一条项链顺时针旋转过$k$颗珠子后和原项链相同，称之为$k$-beautiful。给定$n$和$k$以及$n$个小写字母（珠子颜色），求能组成的$k$-beautiful 的项链最大长度。

从$n$到$1$枚举要制作的项链的长度$len$，对于项链的每个位置$i$，由$i$向$i+k\pmod{len}$连一条边，显而易见这些边可以连成一个或多个环，$cycles[i]$记录每个以$i$起始的环的长度。然后测试手里的珠子能否做出来这样一条项链。

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
	int test;
	cin >> test;

	while (test--) {
		int n, k;
		cin >> n >> k;
		string s;
		cin >> s;

		vector<int> cnt(26);

		for (char c : s) {
			cnt[c - 'a']++;
		}

		for (int len = n; len >= 1; len--) {
			vector<bool> used(len);
			vector<int> cycles;

			for (int i = 0; i < len; i++) {
				if (used[i]) {
					continue;
				}

				int j = (i + k) % len;
				used[i] = true;
				cycles.push_back(0);
				cycles.back()++;

				while (!used[j]) {
					cycles.back()++;
					used[j] = true;
					j = (j + k) % len;
				}
			}

			vector<int> cur_cnt(cnt);

			sort(cycles.begin(), cycles.end());
			sort(cur_cnt.begin(), cur_cnt.end());

			bool can_fill = true;

			while (!cycles.empty()) {
				if (cur_cnt.back() < cycles.back()) {
					can_fill = false;
					break;
				} else {
					cur_cnt.back() -= cycles.back();
					cycles.pop_back();
					sort(cur_cnt.begin(), cur_cnt.end());
				}
			}

			if (can_fill) {
				cout << len << endl;
				break;
			}
		}
	}
}
```

## [1367F - Flying Sort](https://codeforces.com/contest/1367/problem/F2)

挖个坑，等人填。
