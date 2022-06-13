# `Python` 面向对象

## Setter Getter

```python
>>> class Screen(object):
...     def __init__(self):
...         self._width = 800
...         self._height = 600
...     @property
...     def width(self):
...         return self._width
...     @width.setter
...     def width(self, newwidth):
...         self._width = newwidth//2
...
>>> s = Screen()
>>> s.width = 84
>>> print(s.width)
42
```

## Static Method

## Class Method
