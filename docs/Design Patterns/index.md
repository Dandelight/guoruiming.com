# 设计模式

## Creational Patterns 创建型设计模式

### Abstract Factory 抽象工厂模式

> Provide an interface for creating families of related or dependent objects without specifying their concrete classes.

### Builder 建造者模式

> Separate the construction of a complex object from its representation so that the same construction process can create different representations.

### Factory Method 工厂方法模式

> Define an interface for creating an object, but let subclasses decide witch class to instantiate. Factory Method lets a class defer instantiation to subclasses.

### Prototype 原型模式

> Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype.

### Singleton 单例模式

> Ensure a class only has one instance, and provide a global point of access to it.

## Structural Patterns 结构型设计模式

### Adapter 适配器模式

> Convert the interface of a class into another interface clients except. Adapter lets classes work together that couldn't otherwise because of incompatible interfaces.

### Bridge 桥接模式

> Decouple an abstraction from its implementation so that the two can vary independently.

### Composite 组合模式

> Compose objects into tree structures to represent part-whole hierarchies. Composite lets clients treat individual objects and compositions of objects uniformly

### Decorator 装饰器模式

> Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality.

### Facade 门面模式

> Provide a unified interface to a set of interfaces in a subsystem. Facade defines a higher-level interface that makes the subsystem easier to use.

### Flyweight 享元模式

> Use sharing to support large numbers of fine-grained objects efficiently.

### Proxy 代理模式

> Provide a surrogate or placeholder for another object to control access to it.

## Behavioral Patterns 行为型设计模式

### Chain of Responsibility 责任链模式

> Avoid coupling the sender of a request to its receiver by giving more that one chance to handle the request. Chain the receiving objects and pass the request along the chain until an object handles it.

### Command 命令模式

> Encapsulate a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.

### Interpreter 解释器模式

> Given a language, define a representation for its grammar along with an interpreter that uses the representation to interpret sentences in the language.

### Iterator 迭代器模式

> Provide a way to access the elements of a aggregate object sequentially without exposing its underlying representation.

### Mediator 中介者模式

> Define an object that encapsulates how a set of objects interact. Mediator promotes loose coupling by keeping objects from referring to each other explicitly, and it lets you vary their interaction independently.

### Memento 备忘录模式

> Without violation encapsulation, capture and externalize an object's internal state so that the object can be restored to this state later.

### Observer 观察者模式

> Define a one-to-many dependency between objects so that when one object changes state, all its dependents a notified and updated automatically.

### State 状态模式

> Allow an object to alter its behavior when its internal state changes. The object will appear to change its class.

### Strategy 策略模式

> Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.

### Template Method 模板方法模式

> Define the skeleton of an algorithm in an operation, deferring some steps to subclasses. Template Method lets subclasses redefine certain steps of an algorithm without changing the algorithm's structure.

### Visitor 访问者模式

> Represent an operation to be preformed on elements of an object structure. Visitor lets you define a new operation without changing the classes of the elements on which it operates.