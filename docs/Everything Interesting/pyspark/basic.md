# 认识 `PySpark`

`spark` 是是一个围绕速度、易用性和复杂分析构建的大数据处理框架。最初在 2009 年由加州大学伯克利分校的 AMPLab 开发，并于 2010 年成为 Apache 的开源项目之一。

**Spark**扩展了广泛使用的**MapReduce**计算模型，并具有如下优势。

- 首先，Spark 为我们提供了一个**全面、统一**的框架用于管理各种有着不同性质（文本数据、图表数据等）的数据集和数据源（批量数据或实时的流数据）的大数据处理的需求。
- Spark 可以将 Hadoop 集群中的应用在内存中的**运行速度**提升 100 倍，甚至能够将应用在磁盘上的运行速度提升 10 倍。
- Spark 让开发者可以快速的用 **Java、Scala 或 Python** 编写程序。它本身自带了一个超过 80 个高阶操作符集合。而且还可以用它在 shell 中以交互式地查询数据。
- 除了 Map 和 Reduce 操作之外，它还支持 **SQL 查询，流数据，机器学习和图数据**处理。开发者可以在一个数据管道用例中单独使用某一能力或者将这些能力结合在一起使用。

## 2.Hadoop 和 Spark

**Hadoop** 这项大数据处理技术大概已有十年历史，而且被看做是首选的大数据集合处理的解决方案。MapReduce 是一路计算的优秀解决方案，不过对于需要多路计算和算法的用例来说，并非十分高效。数据处理流程中的每一步都需要一个 Map 阶段和一个 Reduce 阶段，而且如果要利用这一解决方案，需要将所有用例都转换成 MapReduce 模式。

**在下一步开始之前，上一步的作业输出数据必须要存储到分布式文件系统中**。因此，复制和磁盘存储会导致这种方式速度变慢。另外 Hadoop 解决方案中通常会包含**难以安装和管理的集群**。而且为了处理不同的大数据用例，还需要集成多种不同的工具（如用于机器学习的 Mahout 和流数据处理的 Storm）。

如果想要完成比较复杂的工作，就必须将一系列的 MapReduce 作业串联起来然后顺序执行这些作业。每一个作业都是高时延的，而且只有在前一个作业完成之后下一个作业才能开始启动。

而 Spark 则允许程序开发者使用有向无环图（[DAG](http://en.wikipedia.org/wiki/Directed_acyclic_graph)）开发复杂的多步数据管道。而且还支持跨有向无环图的内存数据共享，以便不同的作业可以共同处理同一个数据。

Spark 运行在现有的 Hadoop 分布式文件系统基础之上（[HDFS](http://wiki.apache.org/hadoop/HDFS)）提供额外的增强功能。它支持[将 Spark 应用部署到](http://databricks.com/blog/2014/01/21/Spark-and-Hadoop.html)现存的 Hadoop v1 集群（with SIMR – Spark-Inside-MapReduce）或 Hadoop v2 YARN 集群甚至是[Apache Mesos](http://mesos.apache.org/)之中。

我们应该将 **Spark 看作是 Hadoop MapReduce 的一个替代品而不是 Hadoop 的替代品**。其意图并非是替代 Hadoop，而是为了提供一个管理不同的大数据用例和需求的全面且统一的解决方案。

## 3.Spark 特性

**Spark** 通过在数据处理过程中**成本更低的洗牌（Shuffle）**方式，将 MapReduce 提升到一个更高的层次。利用内存数据存储和接近实时的处理能力，Spark 比其他的大数据处理技术的性能要快很多倍。

Spark 还支持大数据查询的延迟计算，这可以帮助优化大数据处理流程中的处理步骤。Spark 还提供高级的 API 以提升开发者的生产力，除此之外还为大数据解决方案提供一致的体系架构模型。

Spark **将中间结果保存在内存中而不是将其写入磁盘**，当需要多次处理同一数据集时，这一点特别实用。Spark 的设计初衷就是既可以在内存中又可以在磁盘上工作的执行引擎。当内存中的数据不适用时，Spark 操作符就会执行外部操作。Spark 可以用于处理大于集群内存容量总和的数据集。

Spark 会尝试在内存中存储尽可能多的数据然后将其写入磁盘。它可以将某个数据集的一部分存入内存而剩余部分存入磁盘。开发者需要根据数据和用例评估对内存的需求。Spark 的性能优势得益于这种内存中的数据存储。

Spark 的其他特性包括：

- 支持比 Map 和 Reduce 更多的函数。
- 优化任意操作算子图（operator graphs）。
- 可以帮助优化整体数据处理流程的大数据查询的延迟计算。
- 提供简明、一致的 Scala，Java 和 Python API。
- 提供交互式 Scala 和 Python Shell。目前暂不支持 Java。

Spark 是用[Scala 程序设计语言](http://www.scala-lang.org/)编写而成，运行于 Java 虚拟机（JVM）环境之上。目前支持如下程序设计语言编写 Spark 应用：

- Scala
- Java
- Python
- Clojure
- R

`Spark` 附加库

`Spark` 在核心库之外，还拥有一系列附加库。这些库包括：

- 集群管理器(Cluster Manager)：Spark 设计为可以高效地在一个计算节点到数千个计算节点之间伸缩计算。为了实现这样的要求，同时获得最大灵活性，Spark 支持在各种集群管理器(Cluster Manager)上运行，目前 Spark 支持 3 种集群管理器:
  - Hadoop YARN(在国内使用最广泛)
  - Apache Mesos(国内使用较少, 国外使用较多)
  - Standalone(Spark 自带的资源调度器, 需要在集群中的每台节点上配置 Spark)
- Spark Core：实现了 Spark 的基本功能，包含任务调度、内存管理、错误恢复、与存储系统交互等模块。SparkCore 中还包含了对弹性分布式数据集(Resilient Distributed Dataset，简称 RDD)的 API 定义。
- **Spark Streaming**：[Spark Streaming](https://spark.apache.org/streaming/)基于微批量方式的计算和处理，可以用于处理实时的流数据。它使用 DStream，简单来说就是一个弹性分布式数据集（RDD）系列，处理实时数据。
- **Spark SQL**：[Spark SQL](https://spark.apache.org/sql/)可以通过 JDBC API 将 Spark 数据集暴露出去，而且还可以用传统的 BI 和可视化工具在 Spark 数据上执行类似 SQL 的查询。用户还可以用 Spark SQL 对不同格式的数据（如 JSON，Parquet 以及数据库等）执行 ETL，将其转化，然后暴露给特定的查询。
- **Spark MLlib**：[MLlib](https://spark.apache.org/mllib/)是一个可扩展的 Spark 机器学习库，由通用的学习算法和工具组成，包括二元分类、线性回归、聚类、协同过滤、梯度下降以及底层优化原语。
- **Spark GraphX**：[GraphX](https://spark.apache.org/graphx/)是用于图计算和并行图计算的新的（alpha）Spark API。通过引入弹性分布式属性图（Resilient Distributed Property Graph），一种顶点和边都带有属性的有向多重图，扩展了 Spark RDD。为了支持图计算，GraphX 暴露了一个基础操作符集合（如 subgraph，joinVertices 和 aggregateMessages）和一个经过优化的 Pregel API 变体。此外，GraphX 还包括一个持续增长的用于简化图分析任务的图算法和构建器集合。
