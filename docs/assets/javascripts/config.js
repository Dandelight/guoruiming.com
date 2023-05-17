// 每次Instant Reload之后重渲染

// MathJax
// 因为 MathJax 的配置问题，必须先于其它脚本加载
window.MathJax = {
  tex: {
    inlineMath: [
      ["\\(", "\\)"],
      ["$", "$"],
    ],
    displayMath: [
      ["\\[", "\\]"],
      ["$$", "$$"],
    ],
    packages: { "[+]": ["boldsymbol"] },
    processEscapes: true,
    processEnvironments: true,
  },
  loader: { load: ["[tex]/boldsymbol"] },
  options: { ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex" },
};

document$.subscribe(() => {
  MathJax.typesetPromise();
});

// highlight.js
document$.subscribe(() => {
  hljs.highlightAll();
});

// mermaid.js
// document$.subscribe(() => {
//   mermaid.init();
// });
