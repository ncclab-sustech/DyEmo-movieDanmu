# DyEmo-movieDanmu
### Crowdsourcing emotion trajectories: Decoding emotion dynamics from Danmu during naturalistic movie viewing.
Emotions are inherently dynamic, yet existing methods fail to capture their fine-grained temporal evolution in naturalistic contexts. Here, we introduce a large language model (LLM)-based framework that decodes high-resolution, multidimensional emotion dynamics from massive-scale crowdsourced Danmu (i.e., bullet-screen comments) during naturalistic movie viewing. We derive continuous ratings across multiple emotion categories from 7.6 million Danmu comments spanning over 100 widely viewed full-length movies, yielding emotion trajectories with second-level temporal resolution. These trajectories align closely with human annotations and demonstrate robustness across various LLM architectures, Danmu sender cohorts, posting-year cohorts, and languages. Leveraging these data, we quantify key dynamical properties—such as inertia, instability, controllability, and self-similarity. Furthermore, we show that the dynamic emotion space is structured around three dimensions—polarity, complexity, and intensity—forming a continuous landscape of mixed emotional states rather than discrete emotion categories. This scalable and ecologically grounded approach provides a powerful framework for understanding emotion dynamics in naturalistic viewing.

![fig1_framework4](https://github.com/user-attachments/assets/31f5e4c5-5496-456f-8ae1-bd662cfac15b)

The Jupyter Notebooks provide code for reproducing figures in this paper:

Figure 1. LLM-based framework for decoding emotion dynamics via crowdsourced Danmu.

Figure 2. Performance of LLMs in Danmu-based emotion decoding.

Figure 3. Alignment of LLM-derived emotion ratings with human ratings based on Danmu and movie viewing.

Figure 4. Reliability of LLM-derived six-dimensional emotion dynamics.

Figure 5. Basic emotion dynamic properties derived from 102 full-length films.

Figure 6. Core dimensions and density distributions of the emotion dynamic space.

Figure 7. Co-occurrence of emotions in naturalistic movies.

### Usage
git clone https://github.com/ncclab-sustech/DyEmo-movieDanmu.git

cd DyEmo-movieDanmu

conda env create -f environment.yml

conda activate your-env-name
