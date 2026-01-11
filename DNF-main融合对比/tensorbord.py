import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# 分析事件文件内容
def analyze_event_file(event_file):
    print(f"分析事件文件: {event_file}")
    print("=" * 50)

    # 获取文件大小
    file_size = os.path.getsize(event_file) / (1024 * 1024)  # MB
    print(f"文件大小: {file_size:.2f} MB")

    # 读取事件
    event_count = 0
    scalar_data = {}
    graph_found = False

    for event in tf.compat.v1.train.summary_iterator(event_file):
        event_count += 1

        if event.HasField('graph_def'):
            graph_found = True
            print("找到图定义数据")

        for value in event.summary.value:
            tag = value.tag
            if value.HasField('simple_value'):
                if tag not in scalar_data:
                    scalar_data[tag] = []
                scalar_data[tag].append((event.step, value.simple_value))

    print(f"总事件数: {event_count}")
    print(f"找到图定义: {graph_found}")
    print(f"标量标签: {list(scalar_data.keys())}")

    # 可视化标量数据（如果有）
    if scalar_data:
        fig, axes = plt.subplots(len(scalar_data), 1, figsize=(10, 4 * len(scalar_data)))
        if len(scalar_data) == 1:
            axes = [axes]

        for i, (tag, data) in enumerate(scalar_data.items()):
            steps, values = zip(*data)
            axes[i].plot(steps, values)
            axes[i].set_title(tag)
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel('Value')

        plt.tight_layout()
        plt.show()

    return graph_found


# 分析你的事件文件
event_file = "/root/DNF-main/runs/CVPR_SONY/baseline/tensorboard/events.out.tfevents.1758719774.autodl-container-45404b8e68-c4343b06.1138.0"
has_graph = analyze_event_file(event_file)

if has_graph:
    print("\n✅ 事件文件中包含计算图数据，可以使用TensorBoard查看")
else:
    print("\n❌ 事件文件中没有找到计算图数据")
    print("可能的原因：")
    print("1. 训练时没有启用图记录（write_graph=True）")
    print("2. 图数据保存在其他事件文件中")
    print("3. 使用的是TensorFlow 2.x的eager execution模式")