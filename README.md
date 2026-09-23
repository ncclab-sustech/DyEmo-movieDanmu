# DyEmo-movieDanmu
### Danmu captures temporal and state-space structure in population-level emotion dynamics.
Emotions are inherently dynamic, yet their fine-grained temporal and multidimensional organization remains difficult to characterize in naturalistic settings. To examine how emotional expression unfolds at the population level, we derived second-level, multidimensional trajectories from 7.57 million temporally anchored Danmu comments spanning 198 hours of naturalistic viewing, with large language models used for semantic inference. The resulting trajectories converged with human judgments and neural responses and remained robust across language models, independent audience samples, posting periods, and linguistic contexts. Different emotion dimensions exhibited distinct temporal signatures in persistence, variability, and transition properties, and these dynamics predicted audience engagement beyond static emotion profiles. When considered jointly, the emotion trajectories occupied a shared low-dimensional space organized by polarity, complexity, and intensity, with states concentrated around a recurrent central basin and varying systematically in activation, blending, and trajectory speed toward the periphery. Together, these findings reveal temporal and state-space organization of population-level emotion dynamics and demonstrate the value of temporally aligned digital traces for observing such dynamics at scale.

![fig1_framework4](https://github.com/user-attachments/assets/31f5e4c5-5496-456f-8ae1-bd662cfac15b)

<img src="demo_video.gif" width="100%">

### We provide code for reproducing figures in this paper:

`reproduce_figure1.ipynb`: Figure 1. LLM-based framework for inferring emotion dynamics via crowdsourced Danmu.

`reproduce_figure2.ipynb`: Figure 2. Performance of LLMs in Danmu-based emotion inference.

`reproduce_figure3.ipynb`: Figure 3. Alignment of LLM-derived emotion ratings with human ratings based on Danmu and movie viewing.

`reproduce_figure4_supfig6_8.ipynb`: Figure 4 and Supplementary Figures 6–8. Reliability of LLM-derived six-dimensional emotion dynamics.

`reproduce_figure5_a-d_supfig9.ipynb`: Figure 5a–d and Supplementary Figure 9. Emotion dynamic properties derived from 102 full-length films.

`reproduce_figure5_e_pred_likes.ipynb`: Figure 5e and Supplementary Figures 15–17. Prediction of audience engagement from emotion dynamics.

`reproduce_figure6_a-c_supfig24.ipynb`: Figure 6a–c and Supplementary Figure 24. Low-dimensional organization and central-basin structure of population-level emotion dynamics.

`reproduce_figure6_d-f.ipynb`: Figure 6d–f. Spatial organization of emotional states around the central basin.

`reproduce_supfig11_nosmooth.ipynb`: Supplementary Figure 11. Effects of smoothing on emotion dynamic properties.

`reproduce_supfig12-14.ipynb`: Supplementary Figures 12–14. Clustering and stability analysis of movie emotion dynamic profiles.

`reproduce_supfig15-17.ipynb`: Supplementary Figures 15–17. Prediction of audience engagement metrics from emotion trajectories and dynamic features.

`reproduce_supfig20_VA.ipynb`: Supplementary Figure 20. Relationship between principal components and LLM-derived valence and arousal ratings.

`reproduce_supfig21_3d.ipynb`: Supplementary Figure 21. Three-dimensional organization of the emotion dynamic space.

`reproduce_supfig22_23_statistics.ipynb`: Supplementary Figures 22–23. Statistics of adaptive window length distributions.

`reproduce_supfig3_27emotions.ipynb`: Supplementary Figure 3. LLM-derived ratings of extended emotion categories from Danmu.

`reproduce_supfig5.ipynb`: Supplementary Figure 5. Reliability comparison between LLM-human and human-human emotion ratings.

`reproduce_supfig7_10.ipynb`: Supplementary Figures 7 and 10. Comparison between adaptive- and fixed-window emotion trajectories and their dynamic properties.

`reproduce_extendedDataFigs_fmri`: Extended Data Figures. Reproduction of fMRI encoding analyses and neural validation results.

`VAD_classification_new`: Code for emotion classification experiments on the VAD dataset.

`Other_LLMs`: Code for emotion inference using additional large language models.

`VLMs`: Code for emotion inference using vision-language models.

### Usage
git clone https://github.com/ncclab-sustech/DyEmo-movieDanmu.git

cd DyEmo-movieDanmu

conda env create -f environment.yml

conda activate your-env-name

### System requirements
The code was run on a Windows 11 operating system. We recommend running the code in an Anaconda environment. See environment.yml for dependencies and versions. The installation typically requires several minutes.




