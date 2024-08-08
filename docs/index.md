---
hide: ["navigation"]
---

# 欢迎回家

站在科技、工程、人文的交叉点，眺望人类的未来。

## 科研

### Multi-View Clustering（2022.9-）

多视图聚类比较简单，只能说作为入门科研的锻炼，也希望早日走出新手村，做点真正用有影响力的研究。

## Project Linear

### 0.0 时代（-2021.4）

设计图是 PS 画的，前端是 Xamarin 写的，但 Xamarin 已经很少有人用了，难以开发。

### 1.0 时代（2021.4-2021.9）

Linear 的 1.0 版本基于微信小程序，当时想得很简单，干就完了，每个人负责一些组件，先写了前端，然后就写后端，勉勉强强可以跑了。

### 1.5 时代（2021.9-2022.12）

微信小程序受到比较大的限制，因此以比较快的速度迁移到 [uni-app](https://dcloud.io/)。这个框架本质是一套小程序平台统一的 API，自己又做了一套类似小程序 API 的运行时，直接作为 APP 安装在手机上——说人话就是 WebView 套壳。之后跌跌撞撞地写 [Vue 2](https://v2.vuejs.org/)，花了大半年时间做了个基本完整的版本开始内测，结果 bug 巨多，而且大多数不是我们自己的 bug 而是框架的“特性”，非常受不了。

### 2.0 时代（2022.12-2023.3）

升级版本有三个原因：

1. 前端混乱，因为当时没有模块化的意识，也没引入 TypeScript，逐渐累积的代码让人抓狂，而 Pinia 等优秀的框架又只在 [Vue3](https://vuejs.org/) 里存在
2. 基于 [Node.js](https://nodejs.org) 和 [Express.js](https://expressjs.com/) 需要手搓很多东西，比如异常处理、token 鉴权（但后来想想，Spring 恐怕是抽象得太多，有点走向另一个极端）
3. 设计师耗时一个月打造了一套全新的，现代化的白色主题 UI

因此开始用 uni-app 再做新版本，结果还是因为经验不足又把代码写烂了。

### 3.0 时代（2023.3-）

我们有幸联系到相似软件的开发者，了解到 Flutter 工具的优势和在大厂中的广泛应用，也认识到 Flutter 代表的高性能前端框架、Spring Boot 代表的标准化开发模式的巨大优势，因此开始用 Flutter + Spring Boot 重写整个服务。

### Action（2015-）

Action 是最早的一款语言 Cosplay 软件，不仅开创了语 C 软件的先河，而且定义了很多“术语”。但不幸的是，后起的逗戏等软件原封不动地爬 Action 的数据，之后入不敷出的 Action 不得不遗憾停服（虽然不久之后逗戏也停服了）。Action 经历了一次短暂重启-停服后，由 2020 年起重新开始内测。代码又悠久的历史，能跑,而因应用市场的要求不得不适配高 API Level，但最大的坎是 Android 10 引入的 [分区存储](https://source.android.com/docs/core/storage/scoped)。当然改完之后发现，Android 10 之后的版本都是小修小补，改 `compileSdkVersion` 和 `targetSdkVersion` 验证一下功能基本就完事了。

其他的问题出在

- 权限管理上，华米 OV 魅又进行了二次开发，二开重灾区就是软件权限。
- 设计模式上，命令式、MVVC 模式、Flutter 都存在，就差 Compose 就真成历史书了。
- 依赖项，依然依赖十年前的 [xUtils](https://source.android.com/docs/core/storage/scoped) 和很多 `jcenter` 中的项目，更新不及时，可能存在暗坑。
- 低版本适配，在 Android 7 上会有渲染进程不能在子进程中运行的报错，但 Android 9 以上都没问题。

### 重大经验教训

1. 技术选型：合适就行，高性能高并发高可用“三高”架构都是被逼出来的，不要在用户只有不到一万人时就试图构思一个百万人的架构，没有真实的业务场景，设计出的架构都是纸上谈兵，不仅对业务没有帮助，反而会拖垮进度
2. 写测试就是以复杂对抗复杂，划分模块、编写测试样例会增加许多代码，但是相比于不增加代码带来的项目本身的不可预测性，这已经很不错了。
3. 有什么不懂的，找个懂的人问问

## 目录

<!-- - Research: My graduate work.
- Engineering
- Technology & Technique: Take-home small tricks.
- Course: Notes for courses both on-site or on-line.
- Life: Things that make me happy.
- WeBook: Books to-be.
- Meta: About this site, its implementation and deployment. -->

- 研究：我的研究生工作
- 工程：改变世界
- 技术：积跬步，成千里
- 课程：学习使我快乐
- 生活：学习之外的快乐
- WeBook：尝试写书
- Meta：关于本站、关于我

> [!IMPORTANT]
>
> 最近在将新项目的点子记录下来，详见：[项目孵化器](./Engineering/index.md)。

## 研究：我的研究生工作

<!-- ![le](media/index/le.gif) -->
<!-- 图：Laplacian Eigenmap 在 MNIST 数据集上的效果可视化 -->

### 科研

- 方向：多模态学习、噪声标签/噪声关联学习
- 导师：<http://pengxi.me/>

* 目前对学术界的唯一贡献是审了一篇稿

## 工程：技术改变世界

### 课余工程

**详见[项目库](./Engineering/index.md)**，招人！长期招人， ~~没工资，~~ 包教技术

- `submit` 作业提交调度平台
- 用于科研论文/代码/数据/笔记管理的 `VS Code` 插件
- 基于 Rust 的跨平台即时通讯底层系统
- Shape of Voice
- Grow: AI 虚拟人陪伴系统
- LISTEN 耳阅读屏：基于多模态模型 Serverless 和端侧推理的视障人士手机屏幕阅读软件
- 基于点云目标检测的机械臂抓取位点预测与动作规划
- [读英语背单词](https://gitee.com/dandelight/readEnglish)

### 已鸽

战略性放弃 ~~（随写随扔的课程作业）~~ ：

- [PaperPeer](https://gitee.com/dandelight/paperpeer)
- [A Silly 2D Game](https://gitee.com/dandelight/starller)

## 技术：积跬步，成千里

### 人工智能与高性能计算

- 语言：**`Python`**、`CUDA`、`C++`、`Rust`
- 数值计算：**`PyTorch`**、`NumPy`、`MATLAB`
- 训练：`Lightning`
- 部署：`ray.io`
- 可视化：`WandB`

### Web 前端

- 语言：**`JavaScript`**、**`TypeScript`**、**`Dart`**、`CSS`、`SASS`
- 框架：**`Flutter`**、`React.js`、`Vue.js`
- 单元测试：**`Jest`**、`Mocha`
- 工程化：`Webpack`、`Vite`

### Web 后端

- 框架：**`Spring Boot`**、`Express.js`、`Nest.js`
- 数据库：**`MongoDB`**、`MySQL`
- 容器：**`Docker`**
- 消息队列：`RabbitMQ`

### 运维与自动化

- Shell：**`PowerShell`**、**`bash`**
- 云安全：**`Cloudflare`**
  <!-- - 配置管理：**`Ansible`**、`SaltStack` -->
  <!-- - 部署：**`Kubernetes`**、`Docker Swarm` -->

### 嵌入式软硬件设计

- 硬件平台：`ESP32`、`Arduino`
- 汇编语言：`MIPS`、`ARM`（的 `move` 指令）
- 嘉立创 EDA（的打开与关闭）

## 课程：学习使我快乐

> 按照熟悉程度排序

- 编译原理
- 计算机网络
- 数据结构
- 操作系统
- 数据库系统原理
- 软件工程

## 生活：学习之外的快乐

偶尔做点小项目

- [dandelight/dandelight](https://github.com/Dandelight/dandelight)
- [dandelight/latex-starter](https://github.com/Dandelight/latex-starter)
- [dandelight/scuthesis](https://github.com/Dandelight/scuthesis)

## WeBook：尝试写书

记在 [WeBook](./WeBook/index.md) 页面。

## Meta：关于本站、关于我

关于本站的构建原理，请见 [Meta](./meta/index.md)；关于本人，请见 [About](./about.md)。

## Friends：友情链接

[WhileBug Peiran Wang](https://whilebug.github.io/)

## GitHub Status

<a href="https://github.com/Dandelight" target="_blank"><img alt="GitHub Heatmap Snake" src="https://raw.githubusercontent.com/Dandelight/dandelight/output/github-snake.svg"></img></a>

<div id="repo-card">正在加载 GitHub commit 信息</div>
<script>
(function () {
  const username = "Dandelight";
  const repo = "dandelight.github.io";

function toUrlEncoded(obj) {
const keyValuePairs = [];
for (const key in obj) {
keyValuePairs.push(
encodeURIComponent(key) + "=" + encodeURIComponent(obj[key])
);
}
return keyValuePairs.join("&");
}
const config = {
"per_page": 5
}
const apiEndpoint = `https://api.github.com/repos/${username}/${repo}/commits?${toUrlEncoded(config)}`;

// Make API request to fetch commits
fetch(apiEndpoint)
.then((response) => response.json())
.then((commits) => {
const recentCommits = commits; // Get the first 5 commits
// Create repository card HTML
let cardHTML = "<h3>Recent Commits</h3><ul>";
recentCommits.forEach((commit) => {
const author = commit.commit.author.name;
const message = commit.commit.message;
const commitURL = commit.html_url;
const commitHash = commit.sha.substring(0, 7);
const localDateTime = new Date(
commit.commit.author.date
).toLocaleString();
cardHTML += `<li>${localDateTime} <a href="${commitURL}" target="_blank"><code>${commitHash}</code> </a>${author}: ${message}</li>`;
});
cardHTML += "</ul>";

      // Display repository card
      document.getElementById("repo-card").innerHTML = cardHTML;
    })
    .catch((error) => {
      document.getElementById("repo-card").innerHTML =
        "加载最近 commit 出错";
      console.error("Error:", error);
    });

})();
</script>

## Gitee Status

<script src='https://gitee.com/dandelight/blog/widget_preview' async defer></script><div id="osc-gitee-widget-tag"></div>
<style>
/* BEGIN Added by GRM */
.osc-gitee-widget-tag li {margin-left:0em;}
.osc_git_box .osc_git_main ul {width: auto;}
/* END Added by GRM */
.osc_pro_color {color: #4183c4 !important;}
.osc_panel_color {background-color: #ffffff !important;}
.osc_background_color {background-color: #ffffff !important;}
.osc_border_color {border-color: #e3e9ed !important;}
.osc_desc_color {color: #666666 !important;}
.osc_link_color * {color: #9b9b9b !important;}
</style>
