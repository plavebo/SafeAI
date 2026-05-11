# SafeAI
## 📚 References & Acknowledgements

This project utilizes the state-of-the-art face analysis toolbox, **InsightFace**. We highly appreciate the authors for open-sourcing their incredible work.

- **Official Project**: [InsightFace GitHub](https://github.com/deepinsight/insightface)
- **Primary Method Used**: ArcFace PyTorch Backbone (`iresnet`)
- **Citation**:
  ```bibtex
  @inproceedings{deng2018arcface,
  title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2019}
  }
  
  @inproceedings{Deng2020CVPR,
  title = {RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild},
  author = {Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
  booktitle = {CVPR},
  year = {2020}
  }
  
  @inproceedings{deng2020subcenter,
    title={Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces},
    author={Deng, Jiankang and Guo, Jia and Liu, Tongliang and Gong, Mingming and Zafeiriou, Stefanos},
    booktitle={Proceedings of the IEEE Conference on European Conference on Computer Vision},
    year={2020}
  }

## 1:N Face Recognition
This project performs 1:N face recognition using a FAISS database built with the ArcFace algorithm.

## File Descrition
### Crawling.ipynb
데이터셋 속 한국 연예인 이미지를 크롤링할 때 사용한 파일입니다.
### FR_w_FAISS


