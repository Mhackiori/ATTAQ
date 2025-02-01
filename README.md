<div id="top"></div>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Mhackiori/ATTAQ">
    <img src="https://i.postimg.cc/VkMP131J/shield.png" alt="Logo" width="150" height="150">
  </a>

  <h1 align="center">ATTAQ</h1>

  <p align="center">
    Adversarial Robustness of Quantum Machine Learning
    <br />
    <a href="https://github.com/Mhackiori/ATTAQ"><strong>Paper in progress »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Mhackiori/ATTAQ">Anonymous Authors</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary><strong>Table of Contents</strong></summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
    </li>
  </ol>
</details>

<div id="abstract"></div>

## 🧩 Abstract

>Quantum Machine Learning (QML) has emerged as a promising paradigm that combines the power of quantum computing with the adaptability of machine learning. QML models have demonstrated potential advantages in solving complex problems across domains such as optimization, pattern recognition, and generative modeling. While much attention has been devoted to exploring their computational benefits and capabilities, the security of QML models and their robustness to adversarial attacks remains an underexplored area in the literature. Understanding adversarial vulnerabilities is critical for assessing the reliability of QML systems, especially as they transition from theoretical constructs to practical applications. In this work, we present **ATTAQ**, a novel framework for evaluating the adversarial robustness of QML models. We investigate the efficacy of adversarial attacks on quantum models and their classical counterparts in different scenarios. We also examine cross-paradigm vulnerabilities by exploring classical attacks on quantum models and vice versa while assessing the effectiveness of adversarial training for QML models. Our analysis reveals that attacks on QML models are notably stronger, with greater effectiveness in white-box scenarios and higher transferability to classical ML models. Additionally, adversarial training offers limited improvement in QML model robustness. Our findings contribute to a deeper understanding of QML security and highlight key considerations for building robust QML systems.

<p align="right"><a href="#top">(back to top)</a></p>
<div id="usage"></div>

## ⚙️ Usage

To train the models, generate the attacks, and evaluate adversarial transferability and adversarial training, start by cloning the repository.

```bash
git clone https://github.com/Mhackiori/ATTAQ.git
cd ATTAQ
```

Then, install the required Python packages by running the following command. We reccomend setting up a dedicated environment to run the experiments.

```bash
pip install -r requirements.txt
```

The framework is based on Torchquantum, which you can install by executing the following commands.

```bash
git clone https://github.com/mit-han-lab/torchquantum.git
cd torchquantum
pip install --editable .
```

<p align="right"><a href="#top">(back to top)</a></p>