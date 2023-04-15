# kotlin 的语法糖(操作符)🍬

作者：\_青\_9609
链接：https://www.jianshu.com/p/c33200857da2
来源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

认识 kotlin 中的 let、with、run、also、apply、map、flatMap 等操作符。

从 java 转到 kotlin 遇到的第一个障碍就是 kotlin 自带的操作符，在看别人代码的时候总是被各种各样的操作符弄的一头雾水。为什么他可以这么写？为什么他可以直接使用对象的属性？这一系列的代码执行之后到底变成了什么样子？伴随着各种各样的问题，我们不得不先学习一下 kotlin 的操作符。

本文对 kotlin 中常用的操作符进行举例说明，方便开发者理解使用。

## 1. 基础操作符

### 1.1 let

> **将调用者传入代码块中，以`it`指代传入的对象，执行代码块的代码，将代码块最后一行结果或 return 指定的数据返回。**
>
> `let`的执行效果和把代码写在代码块外面差不多，主要的作用是可以对`test`变量是否为空做出判断，如果`test`为空则不会执行代码块中的代码。对于代码的阅读性有一定的提升，业务逻辑和临时变量都写在代码块中，方便区分。

```kotlin
test1("test1")

private fun test1(input: String?) {
    // 返回一个字符串中的第一个数字字符所对应的数字，找不到则返回null
    val result = input?.let { // 如果input为空，则不会执行let代码块的代码，直接返回null
        var number: Int? = null
        for (i in it.iterator()) { // 调用传入对象的方法需要使用it引用
            if (Character.isDigit(i)) {
                number = Integer.parseInt(i.toString())
                break
            }
        }
        number
    }
    LogUtil.print("result = $result") // result = 1
}
```

### 1.2 with

> **传入一个对象，在对象内部执行代码块中的代码，可以直接调用传入对象的公共方法及属性，也可以使用`this`指代传入对象进行操作，将代码块最后一行结果或 return 指定的数据返回。**
>
> `with`操作符不好用，他无法以链式调用的方式承接上面的数据，如果传入的对象可能为空，在使用的时候依旧需要对空指针进行判断。一般在需要重复多次调用同一个对象时可以使用这个操作符，可以省去调用对象的名称。

```kotlin
test2("test2")

private fun test2(input: String?) {
    // 返回一个字符串中的第一个数字字符所对应的数字，找不到则返回null
    val result = with(input) {
        if (this == null) {
            return@with null
        }
        var number: Int? = null
        for (i in iterator()) { // 此处可以直接调用String.iterator()方法
            if (Character.isDigit(i)) {
                number = Integer.parseInt(i.toString())
                break
            }
        }
        number
    }
    LogUtil.print("result = $result") // result = 2
}
```

### 1.3 run

> **将调用者传入代码块中，在调用者内部执行代码，可以直接调用传入对象的公共方法及属性，也可以使用`this`指代传入对象进行操作，将代码块最后一行结果或 return 指定的数据返回。**
>
> `run`操作符是`let`和`with`的结合体，将他们的优点集中到一起，既可以插入到链式调用中，又能直接在代码块中调用传入对象的公共方法及属性，而且在调用前进行空指针判断也很方便。

```kotlin
test3("test3")

private fun test3(input: String?) {
    // 返回一个字符串中的第一个数字字符所对应的数字，找不到则返回null
    val result = input?.run { // 如果input为空，则不会执行run代码块的代码，直接返回null
        var number: Int? = null
        for (i in iterator()) { // 此处可以直接调用String.iterator()方法
            if (Character.isDigit(i)) {
                number = Integer.parseInt(i.toString())
                break
            }
        }
        number
    }
    LogUtil.print("result = $result") // result = 3
}
```

### 1.4 also

> **将调用者传入代码块中，以`it`指代传入的对象，执行代码块的代码，代码执行完成后将调用对象返回。**
>
> `also`和`let`的使用方法和执行效果差不多，唯一的区别是`also`返回的是调用者本身。

```kotlin
test4("test4")

private fun test4(input: String?) {
    // 创建一个内容为输入字符串，字号为20sp，颜色为白色的TextView
    val textView = TextView(this)
    val result = textView.also { // 此处可以直接将also连接在构造函数后，能够减少一个临时变量
        it.text = input ?: ""
        it.textSize = 20f
        it.setTextColor(0xFFFFFFFF.toInt()) // 最终的返回值为调用者，并不是最后一行代码的值
    }
    LogUtil.print("result = ${result.text}") // result = test4
}
```

### 1.5 apply

> **将调用者传入代码块中，在调用者内部执行代码，可以直接调用传入对象的公共方法及属性，也可以使用`this`指代传入对象进行操作，执行代码块的代码，代码执行完成后将调用对象返回。**
>
> `apply`是对`also`的升级，调用方法和属性时不用再使用`it`调用。也可以看作是`run`的变种，使用方法和`run`一致，最终返回传入的对象。`apply`常用于设置一个对象的多个属性，对于不支持链式调用的对象，可以提供一个类似链式调用的效果。

```kotlin
test5("test5")

private fun test5(input: String?) {
    // 创建一个内容为输入字符串，字号为20sp，颜色为白色的TextView
    val textView = TextView(this)
    val result = textView.apply { // 此处可以直接将apply连接在构造函数后，能够减少一个临时变量
        text = input ?: ""
        textSize = 20f
        setTextColor(0xFFFFFFFF.toInt()) // 最终的返回值为调用者，并不以最后一行代码的值为准
    }
    LogUtil.print("result = ${result.text}") // result = test5
}
```

### 1.6 forEach & forEachIndexed

> **遍历一个列表，对实现 Iterable 接口的对象进行遍历，将列表中的每一个数据提取出来传递到代码块中，forEach 会将数据用 it 指定并传入代码块中，forEachIndexed 则会多传递一个 index，用于标记当前数据的位置。**
>
> forEach 和 forEachIndexed 并不会返回任何数据

```kotlin
val list = listOf(1, 2, 3, 4, 5)
list.forEach {
    LogUtil.print(it)
}

list.forEachIndexed { index, i ->
    LogUtil.print("$index - $i")
}
```

### 1.7 小结

- **\*以上“在对象内部执行代码”的说法是方便开发者理解，实际的执行位置并不在对象内部，所以只能调用对象的公共方法及属性，但代码书写方式却和在对象内部书写私有方法一样。\***
- `let`、`run`、`apply`、`also`操作符直接写在函数中时，调用者为函数所在对象。
- `let`、`with`、`run`均是以闭包形式执行，返回的数据为 return 数据或最后一行代码的值。
- `apply`、`also`的返回值均是调用者自身。
- 一般情况下使用`run`和`apply`就足以满足业务需求，其他三个操作符了解运行效果，能够读懂别人的代码即可。

## 2. 流程操作符

**以上的基础操作符也可用于流程中的数据处理。**
在 kotlin 之前使用过 RxJava，kotlin 的流程操作符和 RxJava 差不多，在开发过程中可以直接使用 kotlin 内置的操作符而不需要再引入第三方库了。

### 2.1 map

> **一对一的转换，将 n 个数据的列表转换成 n 个数据的列表，类型及数据都可以变换。仅适用于列表或可以转换成列表的数据，准确的说是实现了`kotlin.collections.Iterable<T>`接口的对象(例如：String 会转换成 List<Char>进行处理)。`map`操作符会把列表中的每一个数据提取出来，用`it`指定，然后执行代码块中的代码，返回 return 指定的数据或最后一行代码的值。**
>
> 当我们需要依次处理一个列表中的每个数据的时候就可以使用`map`操作符，相当于 java 的 for-each 循环。和 RxJava 中的 map 效果一样。这个流程对数据的数量不会有影响。

```kotlin
val inputList = listOf(5, 4, 3, 2, 1) //创建一个包含5个数字的列表，类型为List<Int>
val result = inputList.map { // it指代当前处理的数据
    if (it == 1) {
        return@map "first"
    }
    "index_$it"
}
LogUtil.print("result = $result") // result = [index_5, index_4, index_3, index_2, first]
```

示例中输入的是 5 个 int 数字，我们通过判断将值为 1 的数字修改为“first”，其余数字则添加“index\_”前缀，最终输出的是一个字符串数组。建议返回同样类型的数据，这样后续继续处理也会方便一些，如果返回的数据类型不一致，得到的列表类型会是`Any`，不方便继续处理数据。

### 2.2 flatMap

> **一对多的转换，将 n 个数据的列表根据处理逻辑转换成 m 个数据的列表，类型及数据都可以变换。使用要求和方式与`map`一样，但代码块中返回的结果要求是一个列表。最终的结果是将所有返回列表的数据连到一起，组成一个新列表。**
>
> `flatMap`对返回列表的数据个数不做限制，我们可以通过`flatMap`操作符调整列表中数据的个数，也可以将细分的数据提到上层处理。当我们需要把一些对象中的子数据提取到一个列表中时，使用`flatMap`就很方便。

```kotlin
val inputList = listOf(5, 4, 3, 2, 1) //创建一个包含5个数字的列表，类型为List<Int>
val result = inputList.flatMap {
    if (it <= 1) {
        return@flatMap listOf("$it")
    }
    val index: MutableList<String> = mutableListOf()
    for (i in 1..it) { // 这里把传入的数据当作循环次数使用，如果传入数据是个数据模型，也可以直接提取其中的列表数据。
        index.add("$it-$i")
    }
    index
}
LogUtil.print("result = $result")
// result = [5-1, 5-2, 5-3, 5-4, 5-5, 4-1, 4-2, 4-3, 4-4, 3-1, 3-2, 3-3, 2-1, 2-2, 1]
```

示例中的`MutableList`是一个可变列表，kotlin 中分为可变列表和不可变列表，当需要动态修改列表数据个数的时候就要使用可变列表。我们通过`flatMap`操作符对原始列表进行展开处理，最终的结果是将我们每次返回的列表整合成一个新的列表。

### 2.3 use

> **可以自动关闭使用的资源，针对的是实现了`Closeable`接口的数据。`use`操作符会把使用的对象传递到代码块中，用`it`指定，然后执行代码块中的代码，返回 return 指定的数据或最后一行代码的值。**
>
> 使用这个操作符可以代替传统的`try-catch-finally`代码块，而且不会影响流式代码的结构。每次使用需要手动关闭的对象时就可以使用`use`操作符简化代码了。

```kotlin
BufferedReader(InputStreamReader(FileInputStream(File("a.txt")), Charsets.UTF_8)).use {
    val content = it.readLine()
    LogUtil.print(content)
}
```

这个示例展示了一个按行读取文件的效果，先后构建了`File`、`FileInputStream`、`InputStreamReader`、`BufferedReader`，最终通过`use`操作符自动关闭了所有的资源。(`BufferedReader`在关闭的时候会自动关闭引用的`InputStreamReader`，所以对最外层的`BufferedReader`使用`use`即可)。

对于按行读取文件的功能，kotlin 已经提供了相应的扩展方法，该方法也是基于`use`实现的自动关闭功能，直接调用该方法即可。

```kotlin
File("a.txt").readLines().forEach {
    LogUtil.print(it)
}
```

## 3. 总结

kotlin 的各种操作符基本上都是通过扩展方法和内联函数实现的，这些操作符都是为了方便代码开发而添加的，随着 kotlin 越来越成熟，方便开发人员使用的操作符也会越来越多，单纯靠总结现有的操作符是无法全部掌握的，如果遇到不了解的操作符，可以进入操作符的方法中，查看一下源码的实现方式，再配合注释就可以轻松使用大部分操作符了。
