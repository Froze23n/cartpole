# 编译与预览

在本目录使用 MacTeX/TeX Live 的 XeLaTeX 编译两遍：

```bash
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

如果安装后当前终端仍提示找不到 `xelatex`，重新打开终端或把 MacTeX 的 `texbin` 加入 `PATH`。需要含 ctex、Fandol、amsmath、graphicx、hyperref 的 TeX Live 或 MacTeX。图片均位于 figures/，优先使用矢量PDF；实际页数以当前编译结果为准。

若未安装 TeX Live 或 MacTeX，但可使用 Tectonic，也可在本目录运行：

```bash
tectonic main.tex
```

Tectonic 首次编译需要联网获取 LaTeX 宏包和字体；若 `tectonic` 不在 `PATH`，可用本机安装或应用附带的可执行文件路径代替命令名。编译后先核查 PDF 页数、中文正文、公式、表格和图片，再用于提交。某些 Poppler 环境会报 `Missing language pack for 'Adobe-GB1' mapping`，进而漏显 Fandol 中文字形，留下零散的英文和数字；MacTeX 与 Tectonic 编译的 PDF 都可能出现这种预览问题。遇到这种情况，可用能正确显示中文的 PDF 阅读器，或用 Ghostscript/MuPDF 渲染复核。例如，安装了 Ghostscript 且命令可用时，预览首页可运行：

```bash
gs -q -dBATCH -dNOPAUSE -dFirstPage=1 -dLastPage=1 -sDEVICE=png16m -r150 -sOutputFile=preview-page1.png main.pdf
```
