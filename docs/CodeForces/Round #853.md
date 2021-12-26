## [A. Planning](https://codeforces.com/contest/853/problem/A)

> Let's for each moment of time use a plane, which can depart in this moment of time (and didn't depart earlier, of course) with minimal cost of delay.

$$
\text{s.t.} \min \sum_{i=1}^n c_i\cdot(t_i-i) = \sum_{i=1}^n c_i \cdot t_i - \sum_{i=1}^n c_i \cdot i
$$

因为$\sum_{i=1}^n c_i \cdot i$是常数，因此只需要最小化$\sum_{i=1}^n c_i \cdot t_i$。

下面题解证明了贪心解和最优解是等价的

假设最优解中飞机$i$在$b_i$时刻离开，而贪心解中飞机在$a_i$时刻离开。假设$x$是$c_x$最小的飞机，使得$a_x \neq b_x$。贪心的做法每一次都会选取最小的$c_x$，因此$a_x < b_x$；设$y$使得$b_y = a_x$；但是有$c_y \ge b_y$，因此$b_x \cdot c_x + b_y \cdot c_y \ge b_x \cdot c_y + b_y \cdot c_x$，因此，在最优解中对换$b_x$和$b_y$不会损害最优解性质。多次执行此操作之后，对每一个$i$，都可以做到$b_i = a_i$。
