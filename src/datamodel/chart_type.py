from typing import Literal



ChartType1 = [
    "3D-Bar",
    "area_chart",
    "bar_chart",
    # "bar_chart_num",与bar_chart合并
    "box",
    "bubble",
    "candlestick",
    "funnel",
    "heatmap",
    "histogram",
    "line_chart",
    # "line_chart_num",与line_chart合并
    "multi-axes",
    "pie_chart",
    "radar",
    "rings",
    "rose",
    "treemap",
]


ChartType2=[
    # "bar",合并到bar_chart
    # "bar3D",重复
    # "boxplot",等同于box
    "calendar",
    # "candlestick",重复
    "custom",#自定义，存在多种图表类型
    "dataset",#存在多种图表类型
    "dataZoom",#数据集缩放
    # "flowGL",样本只有1个
    # "funnel",重复
    "gauge",#仪表
    # "globe",#全球地貌图表，图表数据肉眼不可见
    # "graph",#有向图或者无向图网络
    # "graphGL",样本只有3个
    # "graphic",#？？？
    # "heatmap",重复
    "line",
    # "lines",样本只有1个，并入lines3D
    # "line3D",样本只有1个，并入lines3D
    # "lines3D",样本只有4个
    # "linesGL",样本只有1个
    # "map",
    # "map3D",样本只有2个
    "parallel",
    # "pictorialBar",
    # "pie",#合并到pie_chart
    # "radar",重复
    # "rich",#样本只有3个
    "sankey",
    "scatter",
    "scatter3D",
    # "scatterGL",样本只有1个
    "sunburst",#旭日图是否就是饼图？
    # "surface",#模型表面图，数据肉眼不可见
    # "themeRiver",样本只有2个
    "tree",#树状网络
    # "treemap",重复
]


CHARTTYPES=[
    "3D-Bar",
    "area_chart",
    "bar_chart",
    "box",
    "bubble",
    "candlestick",
    "funnel",
    "heatmap",
    "histogram",
    "line_chart",
    "multi-axes",
    "pie_chart",
    "radar",
    "rings",
    "rose",
    "treemap",

    "calendar",
    "gauge",
    "parallel",
    "sankey",
    "scatter",
    "scatter3D",
    "sunburst",
    "tree"
]