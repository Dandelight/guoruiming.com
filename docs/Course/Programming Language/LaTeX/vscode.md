# 配置 LaTeX-Workshop

```json
{
  "ltex.additionalRules.enablePickyRules": false,
  "ltex.additionalRules.motherTongue": "zh-CN",
  "ltex.language": "en-US",
  "ltex.checkFrequency": "edit",
  "latex-workshop.latex.tools": [
    {
      "name": "latexmk",
      "command": "latexmk",
      "args": [
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "-pdf",
        "-outdir=%OUTDIR%",
        "%DOC%"
      ],
      "env": {}
    },
    {
      "name": "pdflatex",
      "command": "pdflatex",
      "args": [
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "%DOC%"
      ],
      "env": {}
    }
  ],
  "latex-workshop.latex.recipes": [
    {
      "name": "latexmk",
      "tools": ["latexmk"]
    },
    {
      "name": "PDFLaTeX",
      "tools": ["pdflatex"]
    }
  ],
  "latex-workshop.latex.recipe.default": "lastUsed",
  "latex-workshop.latex.outDir": "build",
  "latex-workshop.synctex.afterBuild.enabled": true,
  "latex-workshop.linting.chktex.enabled": false,
  "latex-workshop.linting.lacheck.enabled": false,
  "latex-workshop.latexindent.args": [
    "-c",
    "%DIR%/",
    "%TMPFILE%",
    "-m",
    "-l",
    "%WORKSPACE_FOLDER%/.latexindent.yaml"
  ],
  "[latex]": {
    "editor.formatOnSave": false, // formatOnSave slows down everything that needs saving
    "editor.wordBasedSuggestions": false
  },
  "files.autoSave": "afterDelay",
  "files.autoSaveDelay": 2500,
  "files.insertFinalNewline": true,
  "editor.wordWrap": "bounded",
  "editor.rulers": [80],
  "files.trimFinalNewlines": true,
  "files.trimTrailingWhitespace": true
}
```

使用了 `LTeX`、`LaTeX Workshop` 在 `VS Code` 中支持 $\mathrm{\LaTeX}$ 语法，基本已经完美，但美中不足的是，`$` 没有自动配对！经过一番探索，除了使用 `Code` 自带的 `code-snippet` 功能（多少有点累赘）外，目前（2022/11/03）可以通过修改 `LaTeX Workshop` 源码中的配置文件实现，之后可能会有原生支持这个功能。

`VS Code` 默认将扩展都放在 `~/.vscode/extensions` 文件夹下（可以通过命令行选项 `--extension-dir` 修改），`LaTeX Workshop` 的文件放在 `.vscode/extensions/james-yu.latex-workshop-8.29.0` 下（版本号可能不同）。在该目录下，找到 `syntax` 目录，里面就是插件对语言的新增定义。打开 ` latex-language-configuration.json` 文件，将 `autoClosingPair` 按照如下方式修改：

```json
{
  // 省略
  "autoClosingPairs": [
    // 省略
    ["{", "}"],
    ["[", "]"],
    ["(", ")"],
    ["`", "'"], // 不要忘了逗号
    ["$", "$"] // 新增
  ]
  // 省略
}
```

也就是在 `autoClosingPairs` 数组中加上 `["$", "$"]` 一项，就可以了。重启 `VS Code`，在 $\rm{\LaTeX}$ 代码中`$` 就可以自动配对了。

废话几句，这个功能已经有人求了五年了[^iss235]。不是技术上不可行，而是这样的话，所有用户的 `$` 都会自动配对且**用户不能取消配对**。用户自己选择 `$` 配对是否生效的权利，是 `VS Code` 的 `API` 提供的。`VS Code` 迟迟没有提供类似的 `API`……

[^iss235]: <https://github.com/James-Yu/LaTeX-Workshop/issues/235>
