# MongoDB Role-based Access Control

官方文档：<https://www.mongodb.com/docs/manual/tutorial/change-own-password-and-custom-data/>

在 `mongosh` 中，可以通过 `db.createUser` 创建用户。创建用户时，MongoDB 采用基于角色的访问控制

```javascript
use admin
db.createUser({
  user: "myadmin1",
  pwd: passwordPrompt(),
  roles: [
    { role: "readWrite", db: "config" },
    "clusterAdmin"
  ]
});
```

其中，权限由**数据库资源**和在指定资源上的**操作**组成。其中，资源可以是数据库、集合、部分集合和集群；操作包括对资源的增删改查。常用的内置角色有：

|                   角色 | 权限                                                                                        |
| ---------------------: | ------------------------------------------------------------------------------------------- |
|                 `read` | 读取指定数据库中的任何数据                                                                  |
|            `readWrite` | `read` + 写指定数据库中任何数据，包括创建、重命名、删除集合                                 |
|              `dbAdmin` | ~~数据库之律者~~ 读取指定数据库并对数据库进行清理、修改、压缩、获取统计信息、执行检查等操作 |
|         `clusterAdmin` | ~~集群之律者~~ 可以对整个集群及数据库系统进行操作                                           |
|            `userAdmin` | ~~人之律者~~ 在指定数据库创建和修改用户                                                     |
|      `readAnyDatabase` | 对除 `config` 和 `local` 之外的数据库拥有 `read` 权限                                       |
| `readWriteAnyDatabase` | 对除 `config` 和 `local` 之外的数据库拥有 `readWrite` 权限                                  |
|   `dbAdminAnyDatabase` | 对除 `config` 和 `local` 之外的数据库拥有 `dbAdmin` 权限                                    |
| `userAdminAnyDatabase` | 对除 `config` 和 `local` 之外的数据库拥有 `userAdmin` 权限                                  |
