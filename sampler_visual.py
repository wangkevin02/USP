import matplotlib as mpl
from matplotlib.colors import LightSource
import os
import torch
import numpy as np
from pathlib import Path
import jsonlines
import re
from tqdm.auto import tqdm
import hashlib
import matplotlib.pyplot as plt
from datetime import datetime
import random
from typing import List, Dict, Optional, Tuple
import time
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download
from umap.umap_ import UMAP
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.preprocessing import RobustScaler

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
random.seed(42)

class EnhancedProfileAnalyzer:
    def __init__(self, 
                 model_name: str = "princeton-nlp/sup-simcse-roberta-large",
                 batch_size: int = 1024,
                 cache_dir: str = "embedding_cache_proportion",
                 model_dir: str = "./model_cache",
                 use_cache: bool = True):
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cache = use_cache

        if not os.path.exists(model_dir):
            snapshot_download(repo_id=model_name, local_dir=model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(self.device)

    def encode_profiles(self, file_path: str) -> Tuple[List[Tuple[str, str]], torch.Tensor]:
        all_profiles = []
        embeddings_list = []
        
        for batch_id, batch in enumerate(read_jsonl_in_batches(file_path)):
            ids, texts = zip(*batch)
            batch_emb = self.encode(texts, batch_id)
            
            all_profiles.extend(batch)
            embeddings_list.append(batch_emb.cpu())
            
        return all_profiles, torch.cat(embeddings_list)

    def analyze_distribution(self, 
                           embeddings: torch.Tensor, 
                           precision: int = 3,
                           n_neighbors: int = 300,
                           min_dist: float = 0.8):
        scaler = RobustScaler()
        embeddings_np = scaler.fit_transform(embeddings.numpy())
        
        reducer = UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=1.2,
            random_state=42,
            densmap=True
        )
        
        umap_result = reducer.fit_transform(embeddings_np)
        points_3d = umap_result[0] if isinstance(umap_result, tuple) else umap_result
        
        if points_3d.shape[1] != 3:
            raise ValueError("UMAP output dimension mismatch")
        
        try:
            kde = gaussian_kde(points_3d.T, bw_method='scott')
        except np.linalg.LinAlgError:
            points_3d += np.random.normal(0, 1e-6, points_3d.shape)
            kde = gaussian_kde(points_3d.T, bw_method='scott')
        
        densities = kde(points_3d.T)
        densities = np.log1p(densities)
        densities = (densities - densities.min()) / (densities.max() - densities.min())
        
        return points_3d, densities

    def create_continuous_terrain(self,
                                points_3d: np.ndarray,
                                densities: np.ndarray,
                                smoothing_sigma: float = 3.0,
                                grid_resolution: int = 1000):
        x_min, x_max = points_3d[:,0].min(), points_3d[:,0].max()
        y_min, y_max = points_3d[:,1].min(), points_3d[:,1].max()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_margin = x_range * 0.1 if x_range > 0 else 0.1
        y_margin = y_range * 0.1 if y_range > 0 else 0.1
        
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min-x_margin, x_max+x_margin, grid_resolution),
            np.linspace(y_min-y_margin, y_max+y_margin, grid_resolution)
        )
        
        valid_mask = ~np.isnan(densities)
        if valid_mask.sum() == 0:
            raise ValueError("All densities are NaN")
        
        try:
            grid_z = griddata(
                points_3d[valid_mask, :2], 
                densities[valid_mask], 
                (grid_x, grid_y), 
                method='cubic', 
                fill_value=np.nanmin(densities)
            )
        except Exception as e:
            print(f"Error during griddata interpolation: {e}")
            grid_z = griddata(
                points_3d[valid_mask, :2], 
                densities[valid_mask], 
                (grid_x, grid_y), 
                method='linear', 
                fill_value=np.nanmin(densities)
            )
        
        grid_z = np.nan_to_num(grid_z, nan=np.nanmin(densities))
        grid_z = np.maximum(grid_z, 0)
        
        sigma = max(1, int(smoothing_sigma * grid_resolution / 500))
        grid_z = gaussian_filter(grid_z, sigma=sigma)
        
        min_positive = np.min(grid_z[grid_z > 0]) if np.any(grid_z > 0) else 1e-6
        grid_z = np.maximum(grid_z, min_positive)
        grid_z = np.power(grid_z, 0.4)

        # 新的绘图方法
        plt.figure(figsize=(20, 15), dpi=300)
        ax = plt.subplot(111, projection='3d')

        # 使用jet颜色映射并直接映射到颜色
        colors = plt.cm.jet((grid_z - grid_z.min()) / (grid_z.max() - grid_z.min()))
        
        # 直接使用plot_surface而不使用LightSource
        surf = ax.plot_surface(
            grid_x, grid_y, grid_z,
            facecolors=colors,
            rstride=5,
            cstride=5,
            linewidth=0,
            antialiased=True,
            alpha=1.0
        )

        # 设置视角
        ax.view_init(elev=35, azim=-45)
        ax.set_box_aspect([1, 1, 0.4])

        # 添加颜色条
        m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
        m.set_array(grid_z)
        plt.colorbar(m, ax=ax, shrink=0.5, aspect=5)

        # 隐藏坐标轴
        ax.set_axis_off()

        # 保存图像
        plt.savefig(
            "optimized_terrain_new.png",
            bbox_inches='tight',
            dpi=300,
            facecolor='white',
            edgecolor='none',
            pad_inches=0
        )
        
        return grid_x, grid_y, grid_z

    def encode(self, texts: List[str], batch_id: int = 0) -> torch.Tensor:
        cache_key = self._generate_cache_key(texts, batch_id)
        
        if self.use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached
            
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                  max_length=512, return_tensors="pt").to(self.device)
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu())
            
            del inputs, outputs
            torch.cuda.empty_cache()
        
        final_emb = torch.cat(embeddings)
        
        if self.use_cache:
            self._save_to_cache(cache_key, final_emb)
        
        return final_emb.to(self.device)

    def _generate_cache_key(self, texts: List[str], batch_id: int) -> str:
        content = "|||".join(texts) + f"||batch_{batch_id}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[torch.Tensor]:
        path = self.cache_dir / f"embedding_cache_{cache_key}.pt"
        if path.exists():
            try:
                data = torch.load(str(path), weights_only=True)
                return data['embeddings'].to(self.device)
            except Exception as e:
                print(f"Cache loading failed: {str(e)}")
                path.unlink(missing_ok=True)
        return None

    def _save_to_cache(self, cache_key: str, embeddings: torch.Tensor):
        path = self.cache_dir / f"embedding_cache_{cache_key}.pt"
        torch.save({'embeddings': embeddings.cpu()}, str(path))

def read_jsonl_in_batches(file_path: str, batch_size: int = 1024):
    with open(file_path, 'r') as f:
        total = sum(1 for _ in f)
    
    with jsonlines.open(file_path) as reader:
        batch = []
        for i, obj in tqdm(enumerate(reader), total=total, desc="Reading"):
            if profile := extract_profile_from_system(obj[0]["content"]):
                batch.append((f"sys_{i}", profile))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

def extract_profile_from_system(text: str) -> Optional[str]:
    pattern = r"your profile is:\s+(.*?)(?=\s+You can say anything you want)"
    try:
        if match := re.search(pattern, text, re.DOTALL):
            profile = match.group(1).strip()
            return profile[:-1] if profile.endswith("..") else profile
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    analyzer = EnhancedProfileAnalyzer()
    
    try:
        all_profiles, embeddings = analyzer.encode_profiles("train.jsonl")
    except torch.cuda.OutOfMemoryError:
        print("GPU内存不足，尝试减小batch_size")
        analyzer.batch_size = 512
        all_profiles, embeddings = analyzer.encode_profiles("train.jsonl")
    
    sample_ratio = 0.2
    if len(embeddings) > 50000:
        indices = torch.randperm(len(embeddings))[:int(len(embeddings)*sample_ratio)]
        embeddings = embeddings[indices]
    
    points_3d, densities = analyzer.analyze_distribution(
        embeddings, 
        n_neighbors=300,
        min_dist=0.8
    )
    
    try:
        analyzer.create_continuous_terrain(
            points_3d, 
            densities,
            grid_resolution=800,
            smoothing_sigma=2.5
        )
    except ValueError as e:
        print(f"可视化失败: {str(e)}")

if __name__ == "__main__":
    main()