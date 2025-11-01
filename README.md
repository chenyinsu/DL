Computational Photography & ISP 学习笔记

Note
本仓库用于系统性地记录个人在计算摄影和图像信号处理领域的学习过程，重点关注经典算法原理、现代ISP管线技术及其硬件协同设计。

📚 目录

#-概述

#-论文研读笔记

    ◦ #经典算法基础-2005-2010
    ◦ #现代isp技术-2020-2023
    ◦ #相关研究扩展

• #-算法复现与实验

• #-资源链接

• #-学习路线图

🏁 概述

本仓库是我个人学习计算摄影和ISP技术的知识库。目标是通过精读从经典到前沿的论文，并辅以必要的代码复现，深入理解以下核心问题：

1.  传统图像处理算法：如去模糊、去雾、图像增强的原理与局限。
2.  现代ISP管线：如何将多个算法集成到端到端的处理流程中。
3.  硬件与算法的协同设计：如硬件在环仿真、可重构ISP等前沿思想。
4.  低功耗与AI驱动的ISP：面向移动端和边缘设备的最新进展。

📖 论文研读笔记
经典算法基础 (2005-2010)
这部分论文奠定了计算摄影许多方向的基石，其思想至今仍被广泛应用。

论文标题  笔记链接  难点与思考

2005年 - Video Enhancement Using Per-Pixel Virtual Exposures notes/2005_virtual_enhancement.md - -

2006年 - Removing Camera Shake from a Single Photograph notes/2006_removing camera shake.md - -

2007年 - Image Deblurring with Blurred/Noisy Image Pairs notes/2007_image deblurring.md - -

2008年 - High-quality Motion Deblurring from a Single Image notes/2008_motion_deblur.md - -

2008年 - Single Image Dehazing  notes/2008_dehaze.md - -

2008年 - Edge-Preserving Decompositions for Multi-Scale Tone and Detail Manipulation notes/2008_edge_preserving.md - -

2010年 - Image Deblurring using Inertial Measurement Sensors notes/2010_imu_deblur.md - -

2015年 - Algorithms for the Enhancement of Dynamic Range 
and Colour Constancy of Digital Images & Video notes/2015_thesis_enhancement.md - -

现代ISP技术 (2020-2023)

这部分论文反映了当前工业界和学术界的研究热点。

论文标题  笔记链接 核心思想摘要 

2020年 - Hardware-in-the-loop End-to-end Optimization of
Camera Image Processing Pipelines  notes/2020_hil.md - -

2021年 - ReconfigISP: Reconfigurable Camera Image Processing Pipeline  notes/2021_reconfig_isp.md - -

2022年 - Abandoning the Bayer-Filter to See in the Dark  notes/2022_abandon_bayer.md - -

2022年 - Neural Photo-Finishing  notes/2022_neural_finishing.md - -

2023年 - DynamicISP: Dynamically Controlled Image Signal Processor  notes/2023_dynamic_isp.md - -

2023年 - Enabling ISPless Low-Power Computer Vision  notes/2023_ispless.md - -
相关研究扩展
论文标题  笔记链接 核心思想摘要 

2022年 - Image-Adaptive YOLO for Object Detection  notes/2022_image_adaptive_yolo.md - -

2022年 - You Only Need 90K Parameters to Adapt notes/2022_90k_parameters.md - -


🧪 算法复现与实验

本部分将尝试复现论文中的核心算法或进行相关的对比实验。

•   项目一：projects/dehaze_dcp/

    ◦   目标： 复现何恺明博士的经典去雾算法。

    ◦   环境： Python, OpenCV, NumPy。

    ◦   进度: ⏳ 计划中

    ◦   projects/dehaze_dcp/code.py | projects/dehaze_dcp/results.md

•   项目二：projects/deblur_pair/

    ◦   目标： 复现2007年使用图像对进行去模糊的算法。

    ◦   环境： Python, OpenCV, NumPy。

    ◦   进度: ⏳ 计划中

    ◦   projects/deblur_pair/code.py | projects/deblur_pair/results.md

(随着学习深入，更多实验项目将在此添加...)

🔗 资源链接

•   重要数据集：



•   相关开源项目：

    ◦   https://github.com/zhoubolei/computational-photography： 计算摄影资源汇总。

•   学习博客：

    ◦   [待添加]

🗺️ 学习路线图

1.  第一阶段（基础）： 精读2005-2010年的经典论文，理解基本问题的数学模型和解决方法。
2.  第二阶段（深化）： 复现1-2个经典算法（如去雾、去模糊），加深对算法细节的理解。
3.  第三阶段（前沿）： 系统阅读2020年后的现代ISP论文，把握技术发展趋势。
4.  第四阶段（实践/总结）： 尝试对不同算法进行对比分析，或提出自己的总结性报告。

©️ 版权说明

本仓库的所有学习笔记和代码均为个人学习目的而创作。

•   论文的版权归原作者和出版商所有。

•   代码复现部分，若参考或修改了现有开源代码，会在对应位置明确注明出处。

开始学习！ 🚀#DL
