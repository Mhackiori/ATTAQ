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

>Quantum Machine Learning (QML) has emerged as a promising paradigm that combines the power of quantum computing with the adaptability of machine learning. QML models have demonstrated potential advantages in solving complex problems across domains such as optimization, pattern recognition, and generative modeling. While much attention has been devoted to exploring their computational benefits and capabilities, the security of QML models and their robustness to adversarial attacks remains an underexplored area in the literature. Understanding adversarial vulnerabilities is critical for assessing the reliability of QML systems, especially as they transition from theoretical constructs to practical applications. In this work, we present an empirical study on the adversarial robustness of QML models. We systematically evaluate the susceptibility of quantum models to adversarial attacks, comparing them with classical counterparts across different settings. Additionally, we examine cross-paradigm attack transferability by testing classical attacks on quantum models and vice versa. Our results indicate that QML models exhibit higher vulnerability in white-box settings and stronger attack transferability than classical ML models. Furthermore, while adversarial training can provide some robustness, its effectiveness remains limited, similar to classical ML models. Our findings contribute to a better understanding of QML security and provide insights into the challenges of building robust QML systems.

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