# 使用 Vue.js 开发 APP

来源：<https://vue-community.org/guide/ecosystem/mobile-apps.html#nativescript-vue-badge-textpopular>

总的来看，国内外多款框架都提供了原生 API 绑定。国外一些应用对 `npm` 生态提供了 first-class 的支持，但国内应用似乎在“去 `npm` 化”，将整个应用拉向自己的一套框架。但 `uni-app` 提供了 `uniCloud` 一系列生态，对于开发还是非常有利的；国外则主要支持 `Google Firebase`，这也是合理的，但不适应国内网络环境。

以下罗列了一些框架，其中 `NativeScript` 是纯原生开发，而其余框架为混合开发。

- `NativeScript`: 首要支持的是一套 `API` 绑定，和 `HTML5+` 异曲同工。支持多种框架，丰富的文档和教程
- `Weex`: 阿里出品，支持 `Vue.js` 和 `React`，但是经历过一段低谷，目前升级 2.0，还处于百废待兴的状态。在阿里内部有所应用
- `React Native`: Facebook 出品，支持 `React`，但并不是太景气的样子，加之 `React` 学习成本高，不推荐直接学习。
- `Vue Native`: Deprecated. No longer maintained.
- `Quasar`: 更像是一个 UI 库附加了个编译到原生的功能。
- `Ionic Vue`: 支持 Angular, Vue, React，但更像是一个组件库增加了编译到原生的功能。

另外还有 Progressive Web Apps，但其内容主要为 `Web` 开发，最多调用一些浏览器的存储，与原生关系不大所以不在这里讨论。
