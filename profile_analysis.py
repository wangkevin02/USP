import os
import time
import hashlib
import torch
import numpy as np
from pathlib import Path
import jsonlines
import re
from tqdm.auto import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.stats import entropy, gaussian_kde
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from matplotlib.legend_handler import HandlerPathCollection
from matplotlib.legend_handler import HandlerLine2D

def calculate_density_metrics(embeddings: np.ndarray) -> Dict[str, float]:
    """优化后的密度指标计算"""
    start_time = time.time()
    
    # 使用近似计算替代全距离矩阵
    n_samples = len(embeddings)
    n_neighbors = min(50, n_samples-1)
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    
    avg_distance = distances.mean()
    local_density = distances.mean(axis=1)
    
    dbscan = DBSCAN(eps=avg_distance, min_samples=5)
    cluster_labels = dbscan.fit_predict(embeddings)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    
    print(f"密度指标计算完成，耗时 {time.time()-start_time:.2f}s")
    return {
        'avg_point_distance': avg_distance,
        'avg_local_density': local_density.mean(),
        'density_variance': np.var(local_density),
        'n_clusters': n_clusters
    }


class DensityBasedProfileAnalyzer:
    def __init__(self, 
                 model_name: str = "princeton-nlp/sup-simcse-roberta-large",
                 batch_size: int = 32,
                 cache_dir: str = "embedding_cache",
                 model_dir: str = "./model_cache"):
        """
        Initialize the analyzer with model and configuration settings.
        """
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model with local cache or download
        if not os.path.exists(model_dir):
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=model_name, local_dir=model_dir)
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate a cache key for embeddings"""
        content_hash = hashlib.md5("||".join(texts).encode()).hexdigest()
        return f"embeddings_{content_hash}"

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        # Check cache first
        cache_key = self._get_cache_key(texts)
        cache_path = self.cache_dir / f"{cache_key}.npy"
        
        # Try to load from cache
        if cache_path.exists():
            return np.load(cache_path)
        
        # If not in cache, generate new embeddings
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            try:
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Save to cache
                np.save(cache_path, embeddings.cpu().numpy())
                
                return embeddings.cpu().numpy()
            except RuntimeError as e:
                print(f"Error generating embeddings: {e}")
                # If out of memory, try processing in smaller batches
                if "CUDA out of memory" in str(e):
                    print("Out of memory. Falling back to CPU or smaller batches.")
                    self.device = torch.device("cpu")
                    self.model = self.model.to(self.device)
                    inputs = inputs.to(self.device)
                    
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    
                    # Save to cache
                    np.save(cache_path, embeddings.cpu().numpy())
                    
                    return embeddings.cpu().numpy()
                raise

    def _extract_profile(self, data: dict) -> Optional[str]:
        """Extract profile text from raw data."""
        pattern = r"your profile is:\s+(.*?)(?=\s+You can say anything you want)"
        if isinstance(data, list) and data:
            for msg in data:
                if msg.get("role") == "system" and (content := msg.get("content")):
                    if match := re.search(pattern, content, re.DOTALL):
                        profile = match.group(1).strip()
                        return profile[:-1] if profile.endswith("..") else profile
        return None

    def load_embeddings(self, embedding_path: str, profiles_path: str) -> Tuple[List[str], np.ndarray]:
        """Load pre-computed embeddings and corresponding profiles."""
        # Load embeddings
        embeddings = np.load(embedding_path)
        
        # Extract profiles from original file
        profiles = []
        with jsonlines.open(profiles_path) as reader:
            for obj in tqdm(reader, desc="Extracting Profiles"):
                if content := self._extract_profile(obj):
                    profiles.append(content)
        
        # Ensure number of profiles matches embeddings
        assert len(profiles) == len(embeddings), "Number of profiles and embeddings must match"
        
        return profiles, embeddings

    def _calculate_uniformity(self, embeddings: np.ndarray) -> float:
        """
        Calculate uniformity using pairwise gaussian kernel.
        L_uniform = log E_{x,y~data} exp(-2||f(x)-f(y)||^2)
        只使用选中的样本计算 uniformity
        """
        # 确保使用选中的样本计算
        n = len(embeddings)
        if n <= 1:
            return 0.0
            
        # 计算选中样本之间的欧氏距离的平方
        dots = np.sum(embeddings**2, axis=1)
        norm_matrix = dots.reshape(-1, 1) + dots.reshape(1, -1) - 2 * np.dot(embeddings, embeddings.T)
        
        # 移除对角线元素（自身与自身的距离）
        mask = ~np.eye(n, dtype=bool)
        norm_matrix = norm_matrix[mask]
        
        # 使用log-sum-exp技巧来避免数值不稳定
        max_norm = np.max(-2 * norm_matrix)
        log_sum_exp = max_norm + np.log(np.sum(np.exp(-2 * norm_matrix - max_norm)))
        
        # 计算均值，注意样本对的数量是 n*(n-1)
        uniformity = log_sum_exp - np.log(n * (n-1))
        
        return uniformity
    
    def _calculate_ldl(self, embeddings: np.ndarray, k: int = 50) -> np.ndarray:
        """
        Calculate Local Density Loss using k-nearest neighbors average distance.
        """
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)  # k+1因为包含点自身
        distances, _ = nbrs.kneighbors(embeddings)
        # 移除距离为0的自身点
        distances = distances[:, 1:]  
        return distances.mean(axis=1)

    def _calculate_group_metrics(self, embeddings: np.ndarray, density_scores: np.ndarray) -> Dict:
        """统一计算群体指标"""
        return {
            'uniformity': self._calculate_uniformity(embeddings),
            'ldl': self._calculate_ldl(embeddings).mean(),
            'density_range': (density_scores.min(), density_scores.max()),
            'size': len(embeddings),
            'density_metrics': calculate_density_metrics(embeddings)
        }

    def cache_density_metrics(self, embeddings: np.ndarray, cache_file: str = "density_metrics_cache.npz") -> Dict:
        """计算并缓存基于k近邻的密度指标"""
        if os.path.exists(cache_file):
            try:
                cache_data = np.load(cache_file, allow_pickle=True)
                print(f"已从缓存加载密度指标：{cache_file}")
                return cache_data['metrics'][()]
            except Exception as e:
                print(f"加载缓存失败: {e}")
                print("重新计算密度指标...")
        
        start_time = time.time()
        
        # 使用k近邻平均距离计算密度
        nbrs = NearestNeighbors(n_neighbors=50).fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        density_scores = 1 / (distances[:, 1:].mean(axis=1))  # 去除自身点，距离越小密度越大
        
        # 计算整体指标
        metrics = {
            'density_scores': density_scores,
            'overall_uniformity': self._calculate_uniformity(embeddings),
            'overall_ldl': self._calculate_ldl(embeddings).mean()
        }
        
        # 保存缓存
        try:
            np.savez(cache_file, metrics=metrics)
            print(f"密度指标已缓存至：{cache_file}")
        except Exception as e:
            print(f"缓存保存失败: {e}")
        
        print(f"密度指标计算完成，耗时 {time.time()-start_time:.2f}s")
        return metrics
    
    def density_based_group_analysis_multi(
            self, 
            embeddings: np.ndarray, 
            profiles: List[str],
            cache_file: str = "density_metrics_cache.npz"
        ) -> Dict:
            """
            使用缓存的密度指标进行多阈值分析
            """
            # 定义采样百分比
            density_percentages = [0.01, 0.05, 0.1, 0.2]
            random_percentages = density_percentages[1:] + [0.5]
            
            # 获取或计算密度指标
            metrics = self.cache_density_metrics(embeddings, cache_file)
            if not metrics:
                print("密度指标计算失败")
                return {}
            
            density_scores = metrics['density_scores']
            overall_uniformity = metrics['overall_uniformity']
            overall_ldl = metrics['overall_ldl']
            
            # 设置随机种子
            np.random.seed(42)
            random_state = np.random.RandomState(42)
            
            all_results = {}
            
            # 对每个百分比进行分析
            for percentage in density_percentages:
                n_samples = int(len(embeddings) * percentage)
                start_time = time.time()
                
                # 密度采样
                majority_indices = np.argsort(density_scores)[-n_samples:]
                minority_indices = np.argsort(density_scores)[:n_samples]
                
                # 随机采样
                if percentage in random_percentages:
                    random_indices = random_state.choice(
                        len(embeddings), 
                        size=n_samples, 
                        replace=False
                    )
                    random_embeddings = embeddings[random_indices]
                    random_metrics = self._calculate_group_metrics(random_embeddings, density_scores[random_indices])
                else:
                    random_metrics = None
                
                # 计算群体指标
                majority_embeddings = embeddings[majority_indices]
                minority_embeddings = embeddings[minority_indices]
                
                results = {
                    'overall': {
                        'uniformity': overall_uniformity,
                        'ldl': overall_ldl
                    },
                    'density_based': {
                        'majority': self._calculate_group_metrics(majority_embeddings, density_scores[majority_indices]),
                        'minority': self._calculate_group_metrics(minority_embeddings, density_scores[minority_indices])
                    }
                }
                
                if random_metrics:
                    results['random_baseline'] = {'group': random_metrics}
                
                all_results[percentage] = results
                print(f"完成 {percentage*100:.1f}% 样本分析，耗时 {time.time()-start_time:.2f}s")
            
            # 添加额外的随机基准百分比
            for percentage in [0.5]:
                if percentage not in density_percentages:
                    start_time = time.time()
                    n_samples = int(len(embeddings) * percentage)
                    random_indices = random_state.choice(
                        len(embeddings), 
                        size=n_samples, 
                        replace=False
                    )
                    random_embeddings = embeddings[random_indices]
                    
                    results = {
                        'overall': {
                            'uniformity': overall_uniformity,
                            'ldl': overall_ldl
                        },
                        'random_baseline': {
                            'group': self._calculate_group_metrics(random_embeddings, density_scores[random_indices])
                        }
                    }
                    all_results[percentage] = results
                    print(f"完成 {percentage*100:.1f}% 随机样本分析，耗时 {time.time()-start_time:.2f}s")
            
            return all_results

    def analyze_random_samples(self, embeddings: np.ndarray, percentage: float) -> Dict:
        """分析随机样本并支持缓存"""
        cache_file = f'random_metrics_cache_{percentage}.npz'
        
        # 尝试从缓存加载
        if os.path.exists(cache_file):
            try:
                cached_data = np.load(cache_file)
                print(f"已从缓存加载 {percentage}% 随机样本分析结果")
                return {
                    'uniformity': float(cached_data['uniformity']),
                    'ldl': float(cached_data['ldl'])
                }
            except Exception as e:
                print(f"加载缓存失败: {str(e)}")
        
        # 计算新的指标
        sample_size = int(len(embeddings) * percentage)
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
        
        metrics = {
            'uniformity': float(self.calculate_uniformity(sample_embeddings)),
            'ldl': float(np.mean(self.calculate_ldl(sample_embeddings)))
        }
        
        # 保存到缓存
        try:
            np.savez(cache_file, 
                     uniformity=metrics['uniformity'],
                     ldl=metrics['ldl'])
            print(f"已缓存 {percentage}% 随机样本分析结果")
        except Exception as e:
            print(f"保存缓存失败: {str(e)}")
        
        return metrics

    def visualize_multi_group_analysis(self, analysis_results: Dict, output_dir: str = "results"):
            """优化版本：修复插值和样式问题"""
            if not analysis_results:
                print("No analysis results to visualize!")
                return
            
            GROUP_CONFIG = {
                'majority': {'color': '#1f77b4', 'marker': 'o', 'label': 'Majority', 'linewidth': 2},
                'minority': {'color': '#ff7f0e', 'marker': 's', 'label': 'Minority', 'linewidth': 2},
                'random': {'color': '#2ca02c', 'marker': '^', 'label': 'Random', 'linewidth': 2},
                'overall': {'color': 'red', 'marker': '*', 'label': 'Overall', 'linewidth': 3}
            }
            
            # 设置字体和样式
            plt.rcParams.update({
                'font.family': 'DejaVu Serif',
                'axes.facecolor': 'white',
                'figure.facecolor': 'white',
                'grid.color': '#dddddd',
                'grid.linewidth': 0.8,
                'axes.edgecolor': 'black',
                'axes.grid': True,
                'grid.alpha': 0.3
            })
            
            # 创建画布
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_facecolor('white')
            
            # 数据预处理
            plot_data = defaultdict(list)
            try:
                for pct in sorted(analysis_results.keys(), key=lambda x: float(x)):
                    res = analysis_results[pct]
                    
                    # 处理整体数据
                    if not plot_data['overall'] and 'overall' in res:
                        plot_data['overall'].append((
                            res['overall']['uniformity'],
                            res['overall']['ldl'],
                            1.0
                        ))
            
                    # 处理density-based组
                    if 'density_based' in res:
                        for group in ['majority', 'minority']:
                            if group in res['density_based']:
                                plot_data[group].append((
                                    res['density_based'][group]['uniformity'],
                                    res['density_based'][group]['ldl'],
                                    float(pct)
                                ))
            
                    # 处理random组
                    if 'random_baseline' in res and 'group' in res['random_baseline']:
                        plot_data['random'].append((
                            res['random_baseline']['group']['uniformity'],
                            res['random_baseline']['group']['ldl'],
                            float(pct)
                        ))
            except Exception as e:
                print(f"数据处理错误: {str(e)}")
                return
            
            # 自定义图例处理器
            class SmallerMarkersHandler(HandlerPathCollection):
                def __init__(self, marker_size=600):  # 增加默认大小
                    self.marker_size = marker_size
                    super().__init__()
                    
                def create_collection(self, *args, **kwargs):
                    collection = super().create_collection(*args, **kwargs)
                    collection.set_sizes([self.marker_size])
                    return collection
            
            # 初始化handler_map
            handler_map = {}
            scatter_objects = {}
            
            # 绘制逻辑
            for group in ['majority', 'minority', 'random', 'overall']:
                points = plot_data.get(group, [])
                if not points:
                    continue
            
                try:
                    # 解包数据
                    x_vals = np.array([p[0] for p in points])
                    y_vals = np.array([p[1] for p in points])
                    pcts = [p[2] for p in points]
            
                    # 绘制散点并保存对象
                    scatter_obj = ax.scatter(
                        x_vals, y_vals,
                        c=GROUP_CONFIG[group]['color'],
                        marker=GROUP_CONFIG[group]['marker'],
                        s=600,
                        edgecolors='black',
                        linewidths=0.8,
                        label=GROUP_CONFIG[group]['label'],
                        zorder=5
                    )
                    
                    # 保存散点对象并设置其handler
                    scatter_objects[group] = scatter_obj
                    handler_map[scatter_obj] = SmallerMarkersHandler(400)  # 增大图例标记大小
            
                    # 添加标注
                    if group != 'overall':
                        for idx, (x, y, p) in enumerate(zip(x_vals, y_vals, pcts)):
                            # 确定标注位置
                            if GROUP_CONFIG[group]['marker'] == '^':  # 对于三角形标记
                                if abs(p - 0.5) < 0.01:  # 50%
                                    xytext = (-20, 0)  # 左侧
                                    ha = 'right'
                                elif abs(p - 0.2) < 0.01:  # 20%
                                    xytext = (20, 0)  # 右侧
                                    ha = 'left'
                                else:
                                    xytext = (0, 20)  # 其他情况放上方
                                    ha = 'center'
                            elif GROUP_CONFIG[group]['marker'] == 's':  # 对于方形标记
                                if abs(p - 0.01) < 0.001:  # 1%
                                    xytext = (-20, 0)  # 左侧
                                    ha = 'right'
                                else:
                                    xytext = (0, 20)  # 其他情况放上方
                                    ha = 'center'
                            else:
                                # 其他标记的点统一放在上方
                                xytext = (0, 20)
                                ha = 'center'
            
                            annotation = ax.annotate(
                                f"{p*100:.0f}%",
                                (x, y),
                                xytext=xytext,
                                textcoords='offset points',
                                fontsize=16,
                                color='black',
                                weight='bold',
                                horizontalalignment=ha,
                                verticalalignment='center',
                                bbox=dict(
                                    facecolor='none',
                                    edgecolor='none',
                                    pad=0
                                ),
                                zorder=6
                            )
            
                except Exception as e:
                    print(f"绘制 {group} 组时发生错误: {str(e)}")
                    continue
            
            # 坐标轴美化
            ax.set_xlabel('Uniformity Loss', fontsize=18, color='black', labelpad=10)
            ax.set_ylabel('Local Density Loss', fontsize=18, color='black', labelpad=10)
            ax.tick_params(axis='both', which='major', labelsize=14, colors='black')
            
            # 图例优化
            legend = ax.legend(
                loc='upper right',
                frameon=True,
                framealpha=0.9,
                edgecolor='#333333',
                fontsize=20,
                title_fontsize=22,
                borderpad=1,
                handletextpad=1.5,
                handler_map=handler_map,
                ncol=1
            )
            legend.get_title().set_color('black')
            
            # 保存输出
            try:
                os.makedirs(output_dir, exist_ok=True)
                plt.tight_layout()
                output_path = os.path.join(output_dir, 'density_analysis.pdf')
                plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
                plt.close()
                print(f"可视化结果已保存至: {output_path}")
            except Exception as e:
                print(f"保存图像时发生错误: {str(e)}")


    def _generate_enhanced_report(self, analysis_results: Dict, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/enhanced_analysis_report.md", "w") as f:
            f.write("# Enhanced Group Analysis Report with Random Baseline\n\n")
            
            # 总体指标
            f.write("## Overall Metrics\n")
            first_result = next(iter(analysis_results.values()))
            f.write(f"- **Uniformity Loss**: {first_result['overall']['uniformity']:.4f}\n")
            f.write(f"- **Local Density Loss**: {first_result['overall']['ldl']:.4f}\n\n")
            
            # 各百分比分析
            for percentage, results in analysis_results.items():
                f.write(f"## {percentage*100:.0f}% Sampling Analysis\n")
                
                # 创建对比表格
                f.write("| Metric | Density Majority | Density Minority | Random Baseline |\n")
                f.write("|--------|------------------|------------------|-----------------|\n")
                
                # 统一提取指标
                metrics = [
                    ('Uniformity Loss', 'uniformity'),
                    ('LDL', 'ldl'),
                    ('Density Range', 'density_range'),
                    ('Sample Size', 'size'),
                    ('Avg Distance', ('density_metrics', 'avg_point_distance')),
                    ('Cluster Count', ('density_metrics', 'n_clusters'))
                ]
                
                for metric_name, key_path in metrics:
                    row = [metric_name]
                    for group in ['density_based.majority', 'density_based.minority', 'random_baseline.group']:
                        keys = group.split('.') + (key_path if isinstance(key_path, tuple) else [key_path])
                        value = results
                        for k in keys:
                            value = value[k]
                        if isinstance(value, float):
                            row.append(f"{value:.4f}")
                        elif isinstance(value, tuple):
                            row.append(f"({value[0]:.2f}, {value[1]:.2f})")
                        else:
                            row.append(str(value))
                    f.write("|" + "|".join(row) + "|\n")
                
                f.write("\n---\n\n")

if __name__ == "__main__":
    analyzer = DensityBasedProfileAnalyzer()
    
    try:
        print("加载数据和嵌入...")
        start_load = time.time()
        profiles, embeddings = analyzer.load_embeddings("train.npy", "train.jsonl")
        print(f"数据加载完成，耗时 {time.time()-start_load:.2f}s")
        
        print("\n执行密度分析...")
        start_analysis = time.time()
        # 使用缓存功能进行分析
        results = analyzer.density_based_group_analysis_multi(
            embeddings, 
            profiles,
            cache_file="density_metrics_cache.npz"  # 添加缓存文件参数
        )
        print(f"分析完成，耗时 {time.time()-start_analysis:.2f}s")
        
        print("\n生成可视化结果...")
        analyzer.visualize_multi_group_analysis(results)
        
    except Exception as e:
        print(f"\n发生错误：{str(e)}")
        if isinstance(e, MemoryError):
            print("内存不足，建议：")
            print("1. 减少数据量")
            print("2. 使用更小的模型")
            print("3. 增加系统内存")
        elif isinstance(e, FileNotFoundError):
            print("文件不存在，请检查：")
            print("1. 数据文件路径是否正确")
            print("2. 文件权限是否正确")
        elif isinstance(e, PermissionError):
            print("权限错误，请检查：")
            print("1. 是否有写入缓存文件的权限")
            print("2. 是否有读取数据文件的权限")

