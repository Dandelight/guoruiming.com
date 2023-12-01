https://stackoverflow.com/a/34549249

Check first if `git config core.quotePath false` helps.

```
git config --global core.quotePath false
```

This is the documentation for `core.quotePath`:

> ```
> core.quotePath
> ```
>
> Commands that output paths (e.g. `ls-files`, `diff`), will quote "unusual" characters in the pathname by enclosing the pathname in double-quotes and escaping those characters with backslashes in the same way C escapes control characters (e.g. `\t` for TAB, `\n` for LF, `\\` for backslash) or bytes with values larger than 0x80 (e.g. octal `\302\265` for "micro" in UTF-8). If this variable is set to false, bytes higher than 0x80 are not considered "unusual" any more. Double-quotes, backslash and control characters are always escaped regardless of the setting of this variable. A simple space character is not considered "unusual". Many commands can output pathnames completely verbatim using the `-z` option. The default value is true.

If not, as stated in "[Git and the Umlaut problem on Mac OS X](https://stackoverflow.com/a/15553796/6309)", try:

```
git config --global core.precomposeunicode true
```

From [`git config` man page](https://git-scm.com/docs/git-config):

> ```
> core.precomposeUnicode
> ```
>
> This option is only used by Mac OS implementation of Git. When `core.precomposeUnicode=true`, Git reverts the unicode decomposition of filenames done by Mac OS. This is useful when sharing a repository between Mac OS and Linux or Windows. (Git for Windows 1.7.10 or higher is needed, or Git under cygwin 1.7). When false, file names are handled fully transparent by Git, which is backward compatible with older versions of Git.
