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

## File Description

### Crawling.ipynb
This file was used to crawl images of Korean celebrities for the dataset.

### FR_w_FAISS.ipynb
This file contains the code for building and utilizing the FAISS database, determining the enrollment set size and threshold, and evaluating performance metrics.

### DB_Enrollment.ipynb
This code allows you to build a FAISS database using your own custom image dataset.

### face_detection
This code aligns faces and resizes images to 112x112 pixels using the InsightFace face detection library.

### rename_dedup.ipynb
This file contains code for refining and deduplicating the crawled image data.

### 데이터_수집.ipynb
This code collects the MS1MV3 and Korean celebrity datasets used to construct the overall dataset, and splits them into the enrollment set and the attack set.

### 데이터셋.zip
This is the dataset used for FAISS database construction and model validation. The directory structure is as follows:

---

## Dataset
Please download and use 데이터셋.zip.
### Directory structure
```
📂 데이터셋/
 ├── 📁 ms1mv3/                           # MS1MV3 글로벌 데이터셋 루트
 │    ├── 📁 identity_0000883/               # 원본 인물 폴더 1 (안에 원본 이미지들)
 │    │    ├── 🖼️ 0000.jpg
 │    │    └── 🖼️ 0001.jpg
 │    ├── 📁 identity_0007883/               # 원본 인물 폴더 2 ...
 │    │    └── 🖼️ 0000.jpg
 │    │
 │    ├── 📁 ms1mv3_Enrollment/           # FAISS DB 지문 등록용 분할 폴더
 │    │    ├── 📁 identity_0000883/          # 등록용 인물 폴더 1 (앞의 N장)
 │    │    │    └── 🖼️ 0000.jpg
 │    │    └── 📁 identity_0007883/          # 등록용 인물 폴더 2 ...
 │    │
 │    └── 📁 ms1mv3_Attack/               # 1:N 식별 성능 검증용 분할 폴더 (나머지 사진들)
 │         ├── 📁 identity_0000883/          # 검증용 인물 폴더 1
 │         │    └── 🖼️ 0002.jpg
 │         └── 📁 identity_0007883/          # 검증용 인물 폴더 2 ...
 │
 └── 📁 한국 연예인_전처리/                  # 한국인 타겟 데이터셋 루트
      ├── 📁 celebrity_A/                 # 원본 인물 폴더 A (예: 김고은 원본 이미지들)
      │    ├── 🖼️ celebrity_A_정면_1_face_0.jpg
      │    └── 🖼️ celebrity_A_옆모습_1_face_0.jpg
      ├── 📁 celebrity_B/                 # 원본 인물 폴더 B (예: 류진 원본 이미지들) ...
      │    └── 🖼️ celebrity_B_정면_1_face_0.jpg
      │
      ├── 📁 한국연예인_Enrollment/          # FAISS DB 지문 등록용 분할 폴더
      │    ├── 📁 celebrity_A/            # 등록용 인물 폴더 A (앞의 N장)
      │    │    └── 🖼️ celebrity_A_옆모습_1_face_0.jpg
      │    └── 📁 celebrity_B/            # 등록용 인물 폴더 B ...
      │
      └── 📁 한국연예인_Attack/              # 1:N 식별 성능 검증용 분할 폴더 (나머지 사진들)
           ├── 📁 celebrity_A/            # 검증용 인물 폴더 A
           │    └── 🖼️ celebrity_A_정면_4_face_0.jpg
           └── 📁 celebrity_B/            # 검증용 인물 폴더 B ...
```

## Model
This project utilizes the IResNet architecture and ArcFace loss function code provided by InsightFace.

---

## Usage
For recognizing specific images, please refer to the FaceRecognition folder.
