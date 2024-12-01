## <div align="center"> [NeurIPS 2024] <i>TripletCLIP </i>: Improving Compositional Reasoning of CLIP via Synthetic Vision-Language Negatives</div>

<div align="center">
  <a href="https://tripletclip.github.io/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=blue&logo=github"></a> &ensp;
  <a href="https://arxiv.org/abs/2411.02545"><img src="https://img.shields.io/static/v1?label=ArXiv&message=2411.02545&color=B31B1B&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/TripletCLIP"><img src="https://img.shields.io/static/v1?label=Data+Models&message=HuggingFace&color=yellow&logo=huggingface"></a> &ensp;

<img src="assets/tripletclip_teaser.png" alt="TripletCLIP" title="" width="50%" />

</div>


This repository will provide access to the dataset, pretrained checkpoints, inference, and training code for our paper, TripletCLIP.
We provide our training scripts written from scratch to train the models reported in paper and OpenCLIP varient for easy reproducibility.

---

## TODOs:

- [x] ~~Release High-Quality Subset of TripletData.~~
- [x] ~~Release all pre-trained and finetuned checkpoints.~~
- [x] ~~Release TripletCLIP adaption on OpenCLIP.~~ [./src/openclip](./src/openclip)
- [ ] Release data generation scripts.
- [ ] Release full TripletData.
- [ ] Release original TripletCLIP training scripts for reproducibility.

## Checkpoints

Below are the checkpoints for the models trained on CC3M and CC12M datasets. The fine-tuning checkpoint is also provided for further customization.

### Table of Checkpoints

| Methods         | CC3M                                                                 | CC12M                                                                |
|------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
| LaCLIP          | [Link](https://huggingface.co/TripletCLIP/CC3M_LaCLIP_ViTB12)          | [Link](https://huggingface.co/TripletCLIP/CC12M_LaCLIP_ViTB12)          |
| LaCLIP+HN       | [Link](https://huggingface.co/TripletCLIP/CC3M_LaCLIP_Real_HN_ViTB12) | -                                                                    |
| NegCLIP         | [Link](https://huggingface.co/TripletCLIP/CC3M_NegCLIP_ViTB12)        | [Link](https://huggingface.co/TripletCLIP/CC12M_NegCLIP_ViTB12)        |
| NegCLIP++       | [Link](https://huggingface.co/TripletCLIP/CC3M_NegCLIPPP_ViTB12)     | [Link](https://huggingface.co/TripletCLIP/CC12M_NegCLIPPP_ViTB12)     |
| TripletCLIP (ours) | [Link](https://huggingface.co/TripletCLIP/CC3M_TripletCLIP_ViTB12) | [Link](https://huggingface.co/TripletCLIP/CC12M_TripletCLIP_ViTB12) |

### Fine-tuning Checkpoint

For fine-tuning based model checkpoint, please refer to the following link:

- [TripletCLIP OpenCLIP Finetuning Checkpoint](https://drive.google.com/file/d/14mupW26LMh6U4FQPa74FOIMEg8MndxCh/view?usp=sharing)



## Citing

If you find the TripletCLIP useful, then consider citing:

```bibtex
@article{patel2024tripletclip,
    author = {Patel, Maitreya and Kusumba, Abhiram and Cheng, Sheng and Kim, Changhoon and Gokhale, Tejas and Baral, Chitta and Yang, Yezhou},
    title = {TripletCLIP: Improving Compositional Reasoning of CLIP via Synthetic Vision-Language Negatives},
    journal={Advances in neural information processing systems},
    year = {2024},
}
```

# Acknowledgement:

We would like to acknowledge the excelletn open-source community OpenCLIP, Huggingface, LAION-AI, and OpenAI for their efforts on making CLIP inference/finetuning and benchmarking easily accessible to all.
