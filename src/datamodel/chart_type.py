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
    # "scatter3D",
    # "scatterGL",样本只有1个
    "sunburst",#旭日图是否就是饼图？
    # "surface",#模型表面图，数据肉眼不可见
    # "themeRiver",样本只有2个
    "tree",#树状网络
    # "treemap",重复
]


CHARTTYPES=[
    "3D-Bar",         # 3D条形图
    "area_chart",     # 面积图
    "bar_chart",      # 条形图
    "box",            # 箱线图
    "bubble",         # 气泡图
    "candlestick",    # 蜡烛图
    "funnel",         # 漏斗图
    "heatmap",        # 热力图
    "histogram",      # 直方图
    "line_chart",     # 折线图
    "multi-axes",     # 多轴图
    "pie_chart",      # 饼图
    "radar",          # 雷达图
    "rings",          # 环图
    "rose",           # 极坐标玫瑰图
    "treemap",        # 树状图
    
    "calendar",       # 日历图
    "gauge",          # 仪表盘
    "parallel",       # 平行坐标图
    "sankey",         # 桑基图
    "scatter",        # 散点图
    "sunburst",       # 太阳burst图
    "tree",           # 树图
    "polar_bar",      # 极坐标条形图
    "gantt",          # 甘特图
    "graph"           # 关系图
]
