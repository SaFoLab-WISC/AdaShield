

<p align="center">
    <img src="assets/AdaShield_logo.jpg" width="150"/>
<p>
<h2 align="center"> <a href="https://arxiv.org/abs/2403.09513">AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>

<h5 align="center">

[![arxiv](https://img.shields.io/badge/Arxiv-2401.15652-red)](https://arxiv.org/pdf/2401.15652.pdf)
![Jailbreak Attacks](https://img.shields.io/badge/Jailbreak-Attacks-yellow.svg?style=plastic)
![Defenses](https://img.shields.io/badge/MultimodalLargeLanguage-Defenses-orange.svg?style=plastic)
![Multimodal Large Language Models](https://img.shields.io/badge/MultimodalLargeLanguage-Models-green.svg?style=plastic)


 The official implementation of our paper "[AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting](https://arxiv.org/abs/2403.09513)", by *[Yu Wang*](https://rain305f.github.io/), [Xiaogeng Liu*](https://sheltonliu-n.github.io), [Yu Li](https://yu-li.github.io/), [Muhao Chen](https://muhaochen.github.io), and [Chaowei Xiao](https://xiaocw11.github.io).*  (* denotes the equal contribution in this work.)






## 📰 News
 We will release all source codes in three weeks.

| Date       | Event    |
|------------|----------|
| **2024/03/15** | 🔥 We have released our paper. Thanks to all collaborators. |


## 💡 Abstract
With the advent and widespread deployment of Multimodal Large Language Models (MLLMs), the imperative to ensure their safety has become increasingly pronounced. However, with the integration of additional modalities, MLLMs are exposed to new vulnerabilities, rendering them prone to structured-based jailbreak attacks, where semantic content (e.g., ``harmful text'') has been injected into the images to mislead MLLMs. In this work, we aim to defend against such threats. Specifically, we propose **Ada**ptive **Shield** Prompting ( **AdaShield** ), which prepends inputs with defense prompts to defend MLLMs against structure-based jailbreak attacks without fine-tuning MLLMs or training additional modules (e.g., post-stage content detector). Initially, we present a manually designed static defense prompt, which thoroughly examines the image and instruction content step by step and specifies response methods to malicious queries. Furthermore, we introduce an adaptive auto-refinement framework, consisting of a target MLLM and a LLM-based defense prompt generator (Defender). These components collaboratively and iteratively communicate to generate a defense prompt. Extensive experiments on the popular structure-based jailbreak attacks and benign datasets show that our methods can consistently improve MLLMs' robustness against structure-based jailbreak attacks without compromising the model's general capabilities evaluated on standard benign tasks.


## ⚡ Framework
<center style="color:#C0C0C0"> 
    <img src="assets/method.png" width="700"/>

AdaShield consists of a defender $D$ and a target MLLM $M$, where $D$ aims to generate the defense prompt $P$ that safeguards $M$ from malicious queries. Then, $P$ is put into $M$ to generate response $R$ for the current malicious query. $D$ uses the previously failed defense prompts and the jailbreak response from $M$ as feedback, and iteratively refines the defense prompt in a chat format.</center>


<center style="color:#C0C0C0"> 
    <img src="assets/example.png" width="700"/>

A conversation example from AdaShield between the target MLLM $M$ and defender $D$. The objective of defender $D$ is to safeguard $M$ from harmful queries for the Sex scenario. $D$ generates the failed prompt to defend against the malicious query for the first time. Then, with the jailbreak response from $M$ and previous defense prompt as feedback, $D$ successfully optimizes defense prompts by injecting the safe rules about the sex scenario, and outputs a reason to elicit interpretability.</center>


## 📑 Citation

Please consider citing 📑 our papers if our repository is helpful to your work, thanks sincerely!

```bibtex
@misc{wang2024adashield,
      title={AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting}, 
      author={Yu Wang and Xiaogeng Liu and Yu Li and Muhao Chen and Chaowei Xiao},
      year={2024},
      eprint={2403.09513},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
