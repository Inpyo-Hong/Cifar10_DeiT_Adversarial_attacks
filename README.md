# Knowledge distillation vulnerability of DeiT through CNN adversarial attack
This repository is the official code for the paper "Knowledge distillation vulnerability of DeiT through CNN adversarial attack" by Inpyo Hong, and Chang Choi. [Neural Computing and Applications]

## Abstract
In the field of computer vision, active research is conducted to improve model performance. The successful application of transformer models in computer vision has led to the development of new models that incorporate this structure. However, the security vulnerabilities of these new models against adversarial attacks have not yet been thoroughly examined. This study investigated the adversarial attack vulnerabilities of DeiT, a model that combines CNN and transformer models through knowledge distillation techniques. We propose that even with only the teacher model (CNN model) information, a fatal attack on DeiT is possible, defining this attack scenario as a partial-white-box environment. In addition, owing to the integration of both CNN’s local information and the transformer’s global information, DeiT is more susceptible to attacks in a black-box environment than other models. The experimental results demonstrate that when adversarial examples (AEs) generated by the teacher model are inserted into DeiT, Fast Gradient Sign Method (FGSM) causes a 46.49% decrease in accuracy, Projected Gradient Descent (PGD) results in a 65.59% decrease. Furthermore, in a black-box environment, AEs generated by ViT and ResNet-50 have detrimental effects on DeiT. Notably, both the CNN and transformer models induced fatal FGSM attacks on DeiT, resulting in vulnerabilities of 70.49% and 53.59%, respectively. These findings demonstrate the additional vulnerability of DeiT to black-box attacks. Moreover, it highlights that DeiT poses a greater risk in practical applications compared to other models. Based on these vulnerabilities, we hope knowledge distillation research with enhanced adversarial robustness will be actively conducted.

![image](https://github.com/user-attachments/assets/fc9ddf3b-cf8d-47bb-9b0a-9dd44773d417)

![image](https://github.com/user-attachments/assets/d0170265-4382-48e0-9a1f-25b8c0791c6f)





## Experimental Results
Top-1 accuracy comparison under partial-white box environment.
![image](https://github.com/user-attachments/assets/d4951189-5976-4041-8bce-fa38e460fced)


## Citation

```
@article{hong2023knowledge,
  title={Knowledge distillation vulnerability of DeiT through CNN adversarial attack},
  author={Hong, Inpyo and Choi, Chang},
  journal={Neural Computing and Applications},
  pages={1--11},
  year={2023},
  publisher={Springer}
}
```
