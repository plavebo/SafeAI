import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from PIL import Image

# 1. InsightFace 깃허브 코드 경로 연동
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSIGHTFACE_PATH = os.path.join(BASE_DIR, 'insightface', 'recognition', 'arcface_torch')

if INSIGHTFACE_PATH not in sys.path:
    sys.path.insert(0, INSIGHTFACE_PATH)

try:
    from backbones import iresnet18, iresnet34, iresnet50, iresnet100
except ImportError:
    raise ImportError(
        f"'{INSIGHTFACE_PATH}'에서 'backbones' 모듈을 찾을 수 없습니다. "
        "insightface 저장소가 올바른 폴더 구조로 클론되어 있는지 확인해주세요."
    )

BACKBONE_FN = {
    'r18':  iresnet18,
    'r34':  iresnet34,
    'r50':  iresnet50,
    'r100': iresnet100,
}

FILE_TO_DEPTH = {
    'arc_r18_fp16_backbone.pth':  'r18',
    'arc_r34_fp16_backbone.pth':  'r34',
    'arc_r50_fp16_backbone.pth':  'r50',
    'arc_r100_fp16_backbone.pth': 'r100',
    'cos_r18_fp16_backbone.pth':  'r18',
    'cos_r34_fp16_backbone.pth':  'r34',
    'cos_r50_fp16_backbone.pth':  'r50',
    'cos_r100_fp16_backbone.pth': 'r100',
}

# ArcFace 표준 이미지 전처리 파이프라인
TRANSFORM = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class FaceEmbedder:
    def __init__(self, weight_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_path = weight_path
        self.model = self._load_model()

    def _load_model(self):
        filename = os.path.basename(self.weight_path)
        if filename not in FILE_TO_DEPTH:
            raise ValueError(f"지원하지 않는 가중치 파일 이름입니다: {filename}")
        
        depth = FILE_TO_DEPTH[filename]
        
        # [수정] fp16=False로 설정하여 모델을 표준 Float32로 생성합니다.
        model = BACKBONE_FN[depth](fp16=False).to(self.device).eval()
        
        sd = torch.load(self.weight_path, map_location=self.device)
        if any(k.startswith('module.') for k in sd):
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
            
        model.load_state_dict(sd, strict=False)
        
        # [수정] 혹시 모를 레이어 타입 불일치 방지를 위해 전 레이어를 float32로 캐스팅합니다.
        model = model.float()
        return model

    def get_image_paths(self, folder_path):
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        for ext in extensions:
            search_pattern = os.path.join(folder_path, '**', ext)
            image_paths.extend(glob.glob(search_pattern, recursive=True))
        return sorted(list(set(image_paths)))

    @torch.no_grad()
    def extract_batch(self, img_paths, batch_size=64):
        all_embs = []
        for i in range(0, len(img_paths), batch_size):
            batch = img_paths[i:i + batch_size]
            tensors = []
            for p in batch:
                try:
                    img = Image.open(p).convert('RGB')
                    tensors.append(TRANSFORM(img))
                except Exception as e:
                    print(f"⚠️ 이미지 로드 실패 ({p}): {e}")
                    continue

            if not tensors:
                continue

            # [수정] 입력 데이터를 확실하게 .float()로 설정하여 FP32로 연산합니다.
            x = torch.stack(tensors).to(self.device).float()
            
            # 모델과 입력 모두 Float 상태임을 확실하게 보장합니다.
            self.model.float()
            
            emb = torch.nn.functional.normalize(self.model(x), dim=1)
            all_embs.append(emb.cpu().float().numpy())

        if not all_embs:
            return np.empty((0, 512), dtype=np.float32)

        return np.vstack(all_embs)
