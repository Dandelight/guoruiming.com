# MATLAB 模块化编程

又名：这还是我认识的那个 MATLAB 吗

本文介绍了 MATLAB 中，有助于模块化编程的内容。

## [`inputParser`](https://ww2.mathworks.cn/help/matlab/ref/inputparser.html)

推荐使用 MATLAB 版本：**R2021a**

**R2021a** 中正式推出了键值对调用。在此之前，学 MATLAB 的同学大多数是这么搞的

```matlab
printPhoto('myfile.jpg','height',10,'width',8)
```

但现在可以直接这样写，

```matlab
printPhoto('myfile.jpg',height=10,width=8)
```

Very Pythonic。

## Function Signature

以往都是通过 <kbd>F1</kbd> 查看提示，通过 <kbd>ctrl</kbd>+<kbd>D</kbd> 查看函数源码。但现在，MATLAB 允许通过 `json` 格式配置函数提示，以实现在函数内

<https://ww2.mathworks.cn/help/matlab/matlab_prog/customize-code-suggestions-and-completions.html>

## [Function Argument Validation](https://ww2.mathworks.cn/help/matlab/matlab_prog/function-argument-validation-1.html)

Introduced in: **R2019b**

看这代码的简洁程度就能激发学习的动力！

```matlab
function ret = example( inputDir, proj, options )
%EXAMPLE An example.
% Do it like this.
% See THEOTHEREXAMPLE.
% See https://stackoverflow.com/a/60178631/20148196
    arguments
        inputDir (1, :) char
        proj (1, 1) projector
        options.foo char {mustBeMember(options.foo, {'bar' 'baz'})} = 'bar'
        options.Angle (1, 1) {double, integer} = 45
        options.Plot (1, 1) logical = false
    end

    % Code always follows 'arguments' block.
    ret = [];
    switch options.foo
        case 'bar'
            ret = sind(options.Angle);
        case 'baz'
            ret = cosd(options.Angle);
    end

    if options.Plot
        plot(proj.x, proj.y)
    end

end
```

可以看到问题的关键，包括变量类型、数组形状、可选参数，都在于 `arguments` 代码块。

参考：<https://ww2.mathworks.cn/help/matlab/ref/arguments.html>。官方文档已经写得很详细了。

## 模块化编程（package）

将一系列函数包含到一个**模块**中，模块是一个 `+` 开头的文件夹，文件夹中有 `.m` 文件。

## 面向对象

面向对象由 `@example` 文件夹、`example.m` 文件组成，`@example` 内的所有文件都是 `example` 这个类的方法。

方法第一个参数是该类的实例，有两种调用方式（与 `GoLang` 十分类似）

- `robot.teach()`
- `teach(robot)`

（实例来自 Robotic Toolbox[^robotics-toolbox]）

<https://ww2.mathworks.cn/company/newsletters/articles/introduction-to-object-oriented-programming-in-matlab.html>

## 其他

### Live Script

一个比较老的特性，和 `Jupyter Notebook` 非常类似，但个人感觉 MATLAB 功能更加强大。

### Implicit Expansion

同 NumPy 的广播（broadcast），从 2016b 开始支持。

## 为啥要折腾这么多

MATLAB 一直被语法问题和工程性不足而饱受诟病[^terrible]，但最近的更新，尤其是近三年的更新，让我看 到了这门语言的发展希望。

作为程序员，切实的好处是，终于不用被 MATLAB 那古老的语法问题而限制手脚了……

毕竟，MATLAB 官方的专业程度恐怕是 `NumPy` 比不上的。虽然 `PyTorch` 能在 Deep Learning 上超过 MATLAB。

## 缺点

MATLAB 尚未计划推出类似于 Python Type Hint[^pep484]，语法级别的支持恐怕是要再等等，不过现在相比以前已经很令人满意了~

[^robotics-toolbox]: Robotics Toolbox, Peter Corke. <https://petercorke.com/toolboxes/robotics-toolbox/>
[^terrible]: _MATLAB is a terrible programming language_, Nikolaus Rath‘s Website. <https://www.rath.org/matlab-is-a-terrible-programming-language.html>
[^pep484]: _PEP 484 -- Type Hints_, Guido van Rossum et al. <https://peps.python.org/pep-0484/>
