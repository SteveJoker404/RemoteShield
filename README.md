<div align="center">

# [RemoteShield: Enable Robust Multimodal Large Language Models for Earth Observation]()



[Rui Min (闵锐)]()
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Liang Yao (姚亮)](https://1e12leon.top/) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Shiyu Miao (缪师宇)]()
<img src="assets/NJU.jpg" alt="Logo" width="15">, &nbsp; &nbsp; 
[Shengxiang Xu (徐圣翔)](https://xushengxianggg.github.io/) 
<img src="assets/SEU.png" alt="Logo" width="15">, &nbsp; &nbsp;

[Yuxuan Liu (刘宇轩)]()
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Chuanyi Zhang (张传一)](https://ai.hhu.edu.cn/2023/0809/c17670a264073/page.htm) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
[Shimin Di (邸世民)](https://cs.seu.edu.cn/shimindi/main.htm) 
<img src="assets/SEU.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Fan Liu (刘凡)](https://multimodality.group/author/%E5%88%98%E5%87%A1/) ✉ 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;


\*  *Equal Contribution*    ✉ *Corresponding Author*


</div>

## Introduction


A robust Multimodal Large Language Model (MLLM) for Earth Observation should possess the cognitive stability to maintain consistent interpretation and reasoning, regardless of the unpredictable perturbations encountered in real-world environments. However, current Remote Sensing MLLMs fundamentally fail to meet this requirement. Trained on carefully curated, high-quality "clean" datasets, they learn brittle mappings that do not generalize to the noisy and shifted conditions of operational Earth Observation. Consequently, their performance degrades when confronted with the noisy, imperfect inputs typical of actual deployment. To quantify and expose this vulnerability, we curate a comprehensive and realistic set of multimodal perturbations. These perturbations simulate environmental visual degradations, such as cloud and fog cover, together with diverse human-centric textual variation ranging from colloquialisms to vague or omitted instructions. Empirical evaluations reveal that these realistic perturbations significantly impair the visual-semantic reasoning capabilities of leading RS foundation models. To this end, we introduce RemoteShield, a robust Remote Sensing MLLM explicitly trained to maintain consistent outputs across realistic input variations. During training, each clean sample is paired with its image-text perturbed variants, forming a semantic equivalence cluster. Rather than directly fitting noisy samples, RemoteShield is optimized through preference learning over clean and perturbed conditions within the same cluster. By comparing model responses to clean and corrupted inputs, the model is encouraged to favor stable responses over perturbation-induced failures. This cross-condition alignment helps the model focus on the underlying task semantics despite visual degradations and textual noise. Experiments on three Earth Observation tasks show that RemoteShield consistently delivers markedly stronger robustness and cross-condition consistency than representative baselines under realistic multimodal perturbations.

## Acknowledge
- Code in this repository is built on [MS-SWIFT](https://github.com/modelscope/ms-swift). We'd like to thank the authors for open sourcing their project.

## Contact
Please Contact ruimin@hhu.edu.cn

