# `latexmk` 及 `VS Code` 配置

`latexmk` 是一个管理  $\mathrm{\LaTeX}$ 编译的程序，使用 `Perl` 语言编写。我的配置如下：

```perl
# Documentation https://texdoc.org/serve/latexmk/0
# Overleaf help https://www.overleaf.com/learn/latex/Articles/How_to_use_latexmkrc_with_Overleaf
# See https://www.latexstudio.net/archives/51493.html


# If zero, do NOT generate a pdf version of the document. If equal to 1, generate a pdf version of
# the document using pdflatex, using the command specified by the $pdflatex variable. If equal to 2,
# generate a pdf version of the document from the ps file, by using the command specified by the
# $ps2pdf variable. If equal to 3, generate a pdf version of the document from the dvi file, by using
# the command specified by the $dvipdf variable. If equal to 4, generate a pdf version of the document
# using lualatex, using the command specified by the $lualatex variable. If equal to 5, generate a
# pdf version (and an xdv version) of the document using xelatex, using the commands specified by the
# $xelatex and xdvipdfmx variables.
# In $pdf_mode=2, it is ensured that .dvi and .ps files are also made. In $pdf_mode=3, it is ensured
# that a .dvi file is also made. But this may be overridden by the document.

$pdf_mode = 5;

$pdflatex = "pdflatex -file-line-error -halt-on-error -interaction=nonstopmode -synctex=1 %O %S";

# xelatex and xdvipdfmx for $pdf_mode=5
$xelatex = "xelatex -file-line-error -halt-on-error -interaction=nonstopmode -no-pdf -synctex=1 %O %S";
$xdvipdfmx = "xdvipdfmx -E -o %D %O %S";

$bibtex_use = 1.5;

# $preview_mode = 1;

# Move aux files to aux.latex and copy *.pdf to the same level as .latexmkrc
# BTW, `aux' is a reserved file name on Windows(R).
$aux_dir = "aux.latex";
$out_dir = ".";

$clean_ext = "hd nav snm synctex.gz xdv";

add_cus_dep('glo', 'gls', 0, 'glo2gls');
sub glo2gls {
    system("makeindex -s gglo.ist -o \"$_[0].gls\" \"$_[0].glo\"");
}
push @generated_exts, "glo", "gls";

add_cus_dep('nlo', 'nls', 0, 'nlo2nls');
sub nlo2nls {
    system("makeindex -s nomencl.ist -o \"$_[0].nls\" \"$_[0].nlo\"");
}
push @generated_exts, "nlo", "nls";

```

VS Code 配置如下：

```json
{
  "[latex]": {
    "editor.formatOnSave": false
  },
  "files.associations": {
    ".latexmkrc": "perl"
  },
  "latex-workshop.latex.autoBuild.run": "onSave",
  "latex-workshop.latex.recipe.default": "lastUsed",
  "latex-workshop.latex.recipes": [
    {
      "name": "latexmk",
      "tools": [
        "latexmk"
      ]
    },
    {
      "name": "XeLaTeX -> BibTeX -> XeLaTeX*2",
      "tools": [
        "xelatex",
        "bibtex",
        "xelatex",
        "xelatex"
      ]
    },
    {
      "name": "PDFLaTeX",
      "tools": [
        "pdflatex"
      ]
    },
    {
      "name": "PDFLaTeX with Shell Escape",
      "tools": [
        "pdflatex-with-shell-escape"
      ]
    },
    {
      "name": "XeLaTeX",
      "tools": [
        "xelatex"
      ]
    },
    {
      "name": "XeLaTeX with Shell Escape",
      "tools": [
        "xelatex-with-shell-escape"
      ]
    },
    {
      "name": "PDFLaTeX -> BibTeX -> PDFLaTeX*2",
      "tools": [
        "pdflatex",
        "bibtex",
        "pdflatex",
        "pdflatex"
      ]
    },
    {
      "name": "BibTeX",
      "tools": [
        "bibtex"
      ]
    },
    {
      "name": "MakeIndex",
      "tools": [
        "makeindex"
      ]
    }
  ],
  "latex-workshop.latex.tools": [
    {
      "args": [
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "-pdf",
        "%DOC%"
      ],
      "command": "latexmk",
      "name": "latexmk"
    },
    {
      "args": [
        "%DOC%"
      ],
      "command": "xelatex",
      "name": "xelatex"
    },
    {
      "args": [
        "--shell-escape",
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "%DOC%"
      ],
      "command": "xelatex",
      "name": "xelatex-with-shell-escape"
    },
    {
      "args": [
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "%DOC%"
      ],
      "command": "pdflatex",
      "name": "pdflatex"
    },
    {
      "args": [
        "--shell-escape",
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "%DOC%"
      ],
      "command": "pdflatex",
      "name": "pdflatex-with-shell-escape"
    },
    {
      "args": [
        "%DOCFILE%"
      ],
      "command": "bibtex",
      "name": "bibtex"
    },
    {
      "args": [
        "%DOCFILE%"
      ],
      "command": "makeindex",
      "name": "makeindex"
    }
  ],
  "latex-workshop.synctex.afterBuild.enabled": true,
  "latex-workshop.view.pdf.invert": 0.9,
  "latex-workshop.view.pdf.invertMode.brightness": 0.7,
  "latex-workshop.view.pdf.invertMode.enabled": "auto",
  "latex-workshop.view.pdf.viewer": "tab",
  "workbench.editorAssociations": {
    "*.pdf": "latex-workshop-pdf-hook"
  }
}
```