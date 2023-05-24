# Hierarchical Integration Diffusion Model for Realistic Image Deblurring

[Zheng Chen](https://scholar.google.com/citations?user=zssRkBAAAAAJ), [Yulun Zhang](http://yulunzhang.com/), [Ding Liu](https://scholar.google.com/citations?user=PGtHUI0AAAAJ&hl=en), [BinXia](https://scholar.google.com/citations?user=rh2fID8AAAAJ), [Jinjin Gu](https://www.jasongt.com/), [Linghe Kong](https://www.cs.sjtu.edu.cn/~linghe.kong/), and [Xin Yuan](https://xygroup6.github.io/xygroup/)

---

> **Abstract:** *Diffusion models (DMs) have recently been introduced in image deblurring and exhibited promising performance, particularly in terms of details reconstruction. However, the diffusion model requires a large number of inference iterations to recover the clean image from pure Gaussian noise, which consumes massive computational resources. Moreover, the distribution synthesized by the diffusion model is often misaligned with the target results, leading to restrictions in distortion-based metrics. To address the above issues, we propose the Hierarchical Integration Diffusion Model (HI-Diff), for realistic image deblurring. Specifically, we perform the DM in a highly compacted latent space to generate the prior feature for the deblurring process. The deblurring process is implemented by a regression-based method to obtain better distortion accuracy. Meanwhile, the highly compact latent space ensures the efficiency of the DM. Furthermore, we design the hierarchical integration module to fuse the prior into the regression-based model from multiple scales, enabling better generalization in complex blurry scenarios. Comprehensive experiments on synthetic and real-world blur datasets demonstrate that our HI-Diff outperforms state-of-the-art methods.* 
>
> <p align="center">
> <img width="800" src="figs/HI-Diff.png">
> </p>

---

|                              GT                              |                            Blurry                            |       [Restormer](https://github.com/swz30/Restormer)        | [Stripformer](https://github.com/pp00704831/Stripformer-ECCV-2022-) |                        HI-Diff (ours)                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="figs/ComS_GT_GOPR0410_11_00-000205.png" height=80/> | <img src="figs/ComS_Blur_GOPR0410_11_00-000205.png" height=80/> | <img src="figs/ComS_Restormer_GOPR0410_11_00-000205.png" height=80/> | <img src="figs/ComS_Stripformer_GOPR0410_11_00-000205.png" height=80/> | <img src="figs/ComS_HI-Diff_GOPR0410_11_00-000205.png" height=80/> |
|      <img src="figs/ComS_GT_scene050-6.png" height=80/>      |     <img src="figs/ComS_Blur_scene050-6.png" height=80/>     |  <img src="figs/ComS_Restormer_scene050-6.png" height=80/>   | <img src="figs/ComS_Stripformer_scene050-6.png" height=80/>  |   <img src="figs/ComS_HI-Diff_scene050-6.png" height=80/>    |

## Results

We achieved state-of-the-art performance on synthetic and real-world blur dataset. Detailed results can be found in the paper.

<details>
<summary>Evaluation on Synthetic Datasets (click to expan)</summary>

- quantitative comparisons in Table 2 of the main paper

<p align="center">
  <img width="900" src="figs/Tab-1.png">
</p>

- visual comparison in Figure 4 of the main paper

<p align="center">
  <img width="900" src="figs/Fig-1.png">
</p>

</details>

<details>
<summary>Evaluation on Real-World Datasets (click to expan)</summary>

- quantitative comparisons in Table 3 of the main paper

<p align="center">
  <img width="900" src="figs/Tab-2.png">
</p>

- visual comparison in Figure 5 of the main paper

<p align="center">
  <img width="900" src="figs/Fig-2.png">
</p>

</details>

## Citation

If you find the code helpful in your resarch or work, please cite the following paper(s).

```
@article{chen2023hierarchical,
  title={Hierarchical Integration Diffusion Model for Realistic Image Deblurring}, 
  author={Chen, Zheng and Zhang, Yulun and Ding, Liu and Bin, Xia and Gu, Jinjin and Kong, Linghe and Yuan, Xin},
  journal={arXiv preprint arXiv:2305.12966},
  year={2023}
}
```

