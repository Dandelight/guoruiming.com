# Installing the `TeX Live` (on `Linux`)

1. Visit the `CTAN archive` on the [USTC Open Source Software Mirror](http://mirrors.ustc.edu.cn/) , Download the http://mirrors.ustc.edu.cn/CTAN/systems/texlive/tlnet/install-tl-unx.tar.gz

2. Unpack the `.tar.gz` file

```zsh
tar -xzvf install-tl-unx.tar.gz
```

3. Run the `install-tl`, which is a `Perl` script.

```zsh
cd install-tl-20201230/
sudo perl installl-tl
```

You can also manually specify the mirror URL:

```zsh
sudo install-tl -repository http://ftp.ctex.org/mirrors/CTAN/systems/texlive/tlnet/
```

4. Wait for the download to finish.

5. Post installation:

Add the following lines to my `~/.zshrc`

```zsh
export PATH=${PATH}:"/usr/local/texlive/2020/bin/x86_64-linux"
export MANPATH="/usr/local/texlive/2020/texmf-dist/doc/man"
export INFOPATH="/usr/local/texlive/2020/texmf-dist/doc/info"
```

6. Run `source ~/.zshrc`, then try it out!

```zsh
latex
man latex
info latex # Note: Deepin doesn't have `info`
```

7. Configure fonts

```zsh
sudo cp texlive-fontconfig.conf /etc/fonts/conf.d/09-texlive.conf
sudo fc-cache -fsv
```

Hundreds of fonts will be available on your computer. Enjoy! (Though it causes a `deepin-wine QQ` font problem)

8. Install [`TeXWorks`](http://www.tug.org/texworks/)

The `Linux` edition of `TeX Live` doesn't contain `TeXworks`, but installing it is fairly easy.

```zsh
sudo apt install texworks
```

9. Other useful or useless tools

   1. [Overleaf](https://www.overleaf.com/): an online, collaborative $\mathrm{\LaTeX}$ editor, with quality $\mathrm{\LaTeX}$ [templates](https://www.overleaf.com/latex/templates) for free download.

   2. [Mathpix](https://mathpix.com/): OCR pick $\mathrm{\LaTeX}$ math formulas.

```zsh
sudo apt install snapd
sudo snap install mathpix-snipping-tool
```

Goodbye 2020!

---

Just one more thing, append the following line to the `/usr/local/texlive/2020/texmf.cnf`
to make sure that `pdfTeX` and `dvipdfmx` can find the right fonts.

```
OSFONTDIR = /usr/share/fonts//;/usr/local/share/fonts//;~/.fonts//
```
