# Matplotlib：使用Python可视化

Matplotlib是一个综合库，用于在Python中创建静态，动画和交互式可视化。

官网 https://matplotlib.org/stable/index.html

示例 https://matplotlib.org/stable/gallery/index.html

Python可视化库Matplotlib绘图入门详解 https://www.sohu.com/a/343708772_120104204

# pyplot

文档 https://matplotlib.org/stable/api/pyplot_summary.html

## pyplot.subplot

```
subplot(nrows, ncols, index, **kwargs)
```

在当前图形上添加一个子图。子图将在具有*nrows*行和*ncols*列的网格上取得 *index*位置。 

*index*从左上角的1开始并向右增加。

*index*也可以是一个二元组，用于指定子图的（*first*， *last*）索引（起始于1，包括*last*）



