最近看 `reStructuredText` 写书很好用，于是想把博客转移到 `sphinx`。首先先把文章转换为 `rst` 格式

```bash
find . -name "*.md" | while read i; do pandoc -f markdown -t rst "output/$i" -o "${i%.*}.rst"; done
```

或者

```shell
find ./ -type f -name "*.md" \
    -exec pandoc \
    --atx-headers \
    --normalize \
    --verbose \
    --wrap=none \
    --toc \
    --reference-links \
    -s -S \
    -t asciidoc -o {}.adoc \;
```
