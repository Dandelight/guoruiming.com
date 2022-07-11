# 技术积累总结

1. 主节点的内存占用很高，在本项目中可以达到 9.3GB，在分配内存时需要分配 12GB；从节点在本项目中的内存占用在 4GB 以下，分配 8GB 以上即可；节点的状态可以通过 Cloudera Manager 用户界面实时查询。
2. `Spark` 是一个灵活可扩展的计算框架，可以使用统一的 API 读写 json、csv、txt 等文本、parquet 等数据仓库和 Hive、jdbc 数据库等多种数据来源，并在单机单线程、单机多线程、YARN 集群等执行环境上高效运行。在本项目中，在开发调试时，我们使用本机读取 `Hive` 中的数据；在生产时可以提交到 YARN 集群上计算。
3. 开发过程中，多表连接查询非常耗费时间，需要考虑是否可以使用聚合函数或窗口函数替代；如果不能，在开发中可以使用 `DataFrame.sample` 方法对数据进行随机采样，降低运算时间。
4. 查询的中间可以适时将中间结果写入 `MySQL`、`Hive` 或文件，减少重复计算。
5. Spark 在启动前的配置中会规定 `driver.memory` 和 `executor.memory`，内存不足可能会因为 `Java` 进程被杀死而执行失败；如果内存中空间不足 Spark 会将内容缓存到磁盘中，如果磁盘空间不足也会导致执行失败。
6. Spark MLlib 中的模型为 Transformer、Estimator 组成的 Pipeline。Transformer 可以使用 `transform` 方法数据进行转换，如 Word2Vec 可以将文本转成词向量；Estimator 可以使用 `fit` 方法进行训练，`fit` 方法返回一个 Model，如 KMeans，SVM 等模型。可以通过 `Model.save` 保存 Model，模型会被保存到 HDFS 中，如果使用相对路径则保存到用户家目录下。
7. 数据可视化可以使用 Zeppelin 进行，但 Zeppelin 内置的可视化形式比较单一，我们使用了 ECharts 进行了部分复杂的可视化任务
