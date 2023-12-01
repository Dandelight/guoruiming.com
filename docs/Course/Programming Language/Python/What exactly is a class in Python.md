# What exactly is a `class`?

```python
>>> class Apple:
...     i = 1 + 2
...     j = i + 1
...
>>> apple = Apple()
>>> apple
<__main__.Apple object at 0x3969ca9d>
>>> apple.i
3
>>> apple.j
4
>>> class Apple:
...    i = 1
...    print("This line is executed")
...
This line is executed
>>> apple = Apple()
>>> apple
<__main__.Apple object at 0x144e2261>
>>> apple.i
1
>>> from random import random
>>> class Apple:
...     i = random()
...     print("This line is executed")
...
This line is executed
>>> a = Apple()
>>> a.i
0.6654069871450241
>>> a = Apple()
>>> a.i
0.6654069871450241
>>> a = Apple()
>>> a.i
0.6654069871450241
```
