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

### Dataset_Collection.ipynb
This code collects the MS1MV3 and Korean celebrity datasets used to construct the overall dataset, and splits them into the enrollment set and the attack set.

### Dataset.zip
This is the dataset used for FAISS database construction and model validation. The directory structure is as follows:

---

## Dataset
Please download and use Dataset.zip.

---

## Model
This project utilizes the IResNet architecture and ArcFace loss function code provided by InsightFace.

---

## Usage
For recognizing specific images, please refer to the FaceRecognition folder.
