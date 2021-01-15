# HandByHandBuildTransformer

本文从自底向上以及自顶两种方法讲述Transformer的构建，当我们想快速理解一个东西的时候，应该抓主去次，来从整体架构上理解，当我们想加深自己对它的理解的时候，可以采用自底向上的方式，来明白每一步具体做了什么！文中我也给了一些帮助理解的课程和文章。本文做了一个中文版的教程，大部分代码来自与[annotated-transformer][https://github.com/harvardnlp/annotated-transformer],希望对你有所帮助。

## 目录树
---|----images                        //所用的图片
   |----paper                         //所涉及到的论文
   |----bottom2top_transformer.ipynb  //自底向上构建transformer
   |----bottom2top_transformer.ipynb  //自顶向下构建transformer
   |----transformer.py                //transformer py文件