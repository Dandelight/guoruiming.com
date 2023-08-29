---
hide: ["navigation"]
---

# 欢迎回家

尘世中的旅行者，欢迎来此小憩。吾辈不才，但愿意将自己积累的知识悉数与君分享。

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
> 最近在将新项目的点子记录下来，详见：[项目孵化器](./Engineering/index)。

## 研究：我的研究生工作

<!-- ![le](media/index/le.gif) -->
<!-- 图：Laplacian Eigenmap 在 MNIST 数据集上的效果可视化 -->

### 科研

- 方向：多模态学习、噪声标签/噪声关联学习
- 导师：<http://pengxi.me/>

* 目前对学术界的唯一贡献是审了一篇稿

## 工程：技术改变世界

### 课余工程

**详见[项目库](./Engineering/index)**，招人！长期招人， ~~没工资，~~ 包教技术

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

- 语言：**`Python`**、**`CUDA`**、`C++`、`Rust`、 ~~`Fortran`~~
- 数值计算：**`PyTorch`**、`NumPy`、`MATLAB`
- 训练：`Lightning`
- 部署：`ray.io`
- 可视化：`WandB`

### Web 前端

- 语言：**`JavaScript`**、**`TypeScript`**、**`Dart`**、`CSS`、`LESS`、
- 框架：**`Flutter`**、`React.js`、`Vue.js`
- 单元测试：**`Jest`**、`Mocha`
- 工程化：`Webpack`、`Vite`

### Web 后端

- 框架：**`Spring Boot`**、`Express.js`
- 数据库：**`MongoDB`**、`MySQL`
- 容器：**`Docker`**、`Vagrant`
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

* [dandelight/dandelight](https://github.com/Dandelight/dandelight)
* [dandelight/latex-starter](https://github.com/Dandelight/latex-starter)
* [dandelight/scuthesis](https://github.com/Dandelight/scuthesis)

## WeBook：尝试写书

记在 [WeBook](./WeBook/index) 页面。

## Meta：关于本站、关于我

关于本站的构建原理，请见 [Meta](./meta/index)；关于本人，请见 [About](./about)。

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
