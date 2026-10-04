# 仿真优化与人工智能 · 网页版

[在线阅读](https://rtkenny.github.io/so-ai-book/)

本仓库保存《Simulation Optimization and Artificial Intelligence》的静态网页版，包含 6 章中文导读、完整原文、参考文献、4 个 Hands-on Lab、实验配图和可下载的 Python 代码。

## 浏览

打开在线阅读地址，或下载整个仓库后用浏览器打开 `index.html`。公式、图片和字体资源保存在本地，无需安装 Node.js 或 Python 即可阅读。

实验脚本位于 `downloads/code/`，使用电脑上的 Python 3.9+ 运行；网页提供实验说明和代码下载。

## 部署与更新

GitHub Pages 从 `main` 分支的根目录发布。仓库根目录的 `.nojekyll` 文件让 GitHub 直接发布静态文件。

更新生成后的 `index.html`、`assets/` 和 `downloads/`，提交并推送到 `main` 后，GitHub Pages 会自动更新网站。LaTeX 原稿和本地构建环境单独维护。

MathJax 的许可证保存在 `assets/vendor/mathjax/LICENSE`。
