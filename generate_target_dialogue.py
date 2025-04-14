import json
import jsonlines
import numpy as np
import torch
import os
import hashlib
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
from matplotlib.colors import TABLEAU_COLORS
from scipy.stats import gaussian_kde, mannwhitneyu
import matplotlib.lines as mlines

class DialogueAnalyzer:
    def __init__(
        self, 
        model_name: str = "princeton-nlp/sup-simcse-roberta-large",
        batch_size: int = 32,
        cache_dir: str = "embedding_cache",
        model_dir: str = "./model_cache"
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        print("⏳ Loading model...")
        start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(self.device)
        print(f"✅ Model loaded | Time: {time.time()-start:.1f}s")
        self.model.eval()
        os.makedirs(cache_dir, exist_ok=True)

    def _get_dialogue_text(self, dialogue):
        """Concatenate all text content in the dialogue"""
        return " ".join([turn["content"] for turn in dialogue if turn["role"] != "system"])

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _get_cache_path(self, text):
        """Generate cache file path"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{text_hash}.npy")

    def _get_cached_embedding(self, text):
        """Get embedding from cache"""
        cache_path = self._get_cache_path(text)
        if os.path.exists(cache_path):
            return np.load(cache_path)
        return None

    def _save_to_cache(self, text, embedding):
        """Save embedding to cache"""
        cache_path = self._get_cache_path(text)
        np.save(cache_path, embedding)

    def get_embeddings(self, texts):
        """Generate text embeddings in batch"""
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch = texts[i:i+self.batch_size]
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.concatenate(embeddings)

    def get_embeddings_with_cache(self, texts):
        """Generate embeddings with caching"""
        embeddings = []
        texts_to_process = []
        indices_to_process = []

        for i, text in enumerate(texts):
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                texts_to_process.append(text)
                indices_to_process.append(i)

        if texts_to_process:
            new_embeddings = self.get_embeddings(texts_to_process)
            
            for text, embedding in zip(texts_to_process, new_embeddings):
                self._save_to_cache(text, embedding)
            
            for idx, embedding in zip(indices_to_process, new_embeddings):
                embeddings.insert(idx, embedding)

        return np.array(embeddings)

    def analyze_dialogues(self, file_path):
        """修改后的分析方法，使用二维PCA"""
        target_dialogues = []
        generated_dialogues = {}
        
        with jsonlines.open(file_path) as reader:
            all_models = set()
            for obj in reader:
                if "models" in obj:
                    all_models.update(obj["models"].keys())
            reader.close()
            
            for model in all_models:
                generated_dialogues[model] = []
            
            with jsonlines.open(file_path) as reader:
                for obj in tqdm(reader, desc="Reading data"):
                    if "target_dialogue" in obj:
                        target_text = self._get_dialogue_text(obj["target_dialogue"])
                        if target_text.strip():
                            target_dialogues.append(target_text)
                    
                    if "models" in obj:
                        for model_name in all_models:
                            if model_name in obj["models"]:
                                dialogue = obj["models"][model_name]
                                text = self._get_dialogue_text(dialogue)
                                generated_dialogues[model_name].append(text if text.strip() else None)

        print(f"\nFound {len(target_dialogues)} target dialogues")
        target_embeddings = self.get_embeddings_with_cache([d for d in target_dialogues if d])
        
        generated_embeddings = {}
        for model_name, dialogues in generated_dialogues.items():
            valid_dialogues = [d for d in dialogues if d]
            if valid_dialogues:
                embeddings = self.get_embeddings_with_cache(valid_dialogues)
                generated_embeddings[model_name] = embeddings

        # 二维PCA降维
        all_embeddings = np.concatenate([target_embeddings] + [e for e in generated_embeddings.values()])
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(all_embeddings)
        
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(scaled_embeddings)
        
        # 分割结果
        target_pca = pca_embeddings[:len(target_embeddings)]
        pca_results = {}
        start = len(target_embeddings)
        for model_name in generated_embeddings:
            end = start + len(generated_embeddings[model_name])
            pca_results[model_name] = pca_embeddings[start:end]
            start = end

        return {
            'target_embeddings': target_pca,
            'generated_embeddings': pca_results
        }

    
    def visualize_2d_differences(self, analysis_results, output_file="2d_diff_visualization.pdf"):
        """
        优化版的二维差异可视化，选择60%位置的最小值点进行标注
        """
        if not analysis_results:
            print("No analysis results to visualize!")
            return
    
        # 设置字体
        plt.rcParams['font.family'] = 'DejaVu Serif'
        
        plt.figure(figsize=(16, 10))
        
        # 配置颜色和样式
        colors = plt.cm.tab10.colors
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
        marker_styles = ['o', 's', '^', 'D', 'P']
        
        # 创建坐标轴
        ax = plt.gca()
        
        # 存储所有模型在60%位置的值
        values_at_60 = {}
        all_curves = {}
        
        # 定义期望的模型顺序
        desired_order = ['ProfileGPT(4o)', 'DialogueGPT(4o)', 'PlatoLM', 'USP w/o RLCC', 'USP']
        
        # 模型名称映射
        model_name_map = {
            'gpt4o_profile': 'ProfileGPT(4o)',
            'gpt4o_dialoge': 'DialogueGPT(4o)',
            'plato': 'PlatoLM',
            'ppo': 'USP',
            'us_without_ppo': 'USP w/o RLCC'
        }
        
        # 反向映射，用于找到对应的原始键名
        reverse_map = {v: k for k, v in model_name_map.items()}
        
        # 预处理数据并绘制主曲线
        max_x = 0
        
        # 按照期望顺序绘制曲线
        for model_idx, desired_name in enumerate(desired_order):
            # 获取原始模型名称
            original_name = reverse_map[desired_name]
            
            # 检查模型是否存在于数据中
            if original_name not in analysis_results['generated_embeddings']:
                continue
                
            embeddings = analysis_results['generated_embeddings'][original_name]
            
            # 计算绝对差异
            target = analysis_results['target_embeddings'][:len(embeddings)]
            differences = np.linalg.norm(embeddings - target, axis=1)
            sorted_diff = np.sort(np.abs(differences))
            
            # 计算分位点（每5%一个点）
            percentiles = np.linspace(0, 100, 21)
            quantiles = np.percentile(sorted_diff, percentiles)
            
            # 存储曲线数据
            all_curves[desired_name] = (percentiles, quantiles)
            
            # 找到最接近60%的值
            idx_60 = np.abs(percentiles - 60).argmin()
            values_at_60[desired_name] = quantiles[idx_60]
            
            # 绘制主曲线
            ax.plot(percentiles, quantiles, 
                    color=colors[model_idx],
                    linestyle=line_styles[model_idx % len(line_styles)],
                    marker=marker_styles[model_idx % len(marker_styles)],
                    markersize=8,
                    linewidth=2.0,
                    label=desired_name)
            
            max_x = max(max_x, np.max(sorted_diff))
        
        # 找到60%位置的最小值
        max_model = max(values_at_60.items(), key=lambda x: x[1])
        max_model_name = max_model[0]
        max_value = max_model[1]
        
        # 添加局部参考线（只在点的左侧和下侧）
        # 垂直参考线（从最小值点到x轴）
        ax.plot([60, 60], [0, max_value], 
                color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # 水平参考线（从最小值点到y轴）
        ax.plot([0, 60], [max_value, max_value], 
                color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # 使用星形标记替代scatter点和标注
        ax.plot(60, max_value, marker='X', color='red', markersize=25, zorder=5)
        
        # 坐标轴设置
        ax.set_xlabel('Percentile (%)', fontsize=26)
        ax.set_ylabel('Absolute Difference Value', fontsize=26)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, max_x*1.1)
        
        # 增大刻度字体
        ax.tick_params(axis='both', labelsize=22)
        
        # 创建图例
        import matplotlib.lines as mlines
        
        handles = []
        for idx, model_name in enumerate(desired_order):
            line = mlines.Line2D([], [], 
                               color=colors[idx],
                               linestyle=line_styles[idx % len(line_styles)],
                               marker=marker_styles[idx % len(marker_styles)],
                               markersize=8,
                               linewidth=2.0,
                               label=model_name)
            handles.append(line)
        
        ax.legend(handles=handles,
                 loc='upper left',
                 fontsize=22,
                 frameon=True,
                 framealpha=0.9,
                 edgecolor='lightgray',
                 ncol=1)
        
        # 优化布局
        plt.tight_layout()
        
        # 保存为PDF
        plt.savefig(output_file, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_file}")


    def _compute_kl_divergence(self, P, Q):
        """
        Compute KL divergence between two distributions P and Q
        Using a small epsilon to avoid division by zero
        """
        epsilon = 1e-10
        P = P + epsilon
        Q = Q + epsilon
        
        # Normalize to make sure they sum to 1
        P = P / np.sum(P)
        Q = Q / np.sum(Q)
        
        return np.sum(P * np.log(P / Q))
    
    def _estimate_density(self, points, bins=20):
        """
        Estimate the probability density of points in 3D space
        using histogram binning
        """
        H, edges = np.histogramdd(points, bins=bins)
        return H / np.sum(H)
    
    def generate_report(self, analysis_results, output_file="analysis_report.txt"):
        """简化的报告生成方法"""
        if not analysis_results:
            print("No valid analysis results!")
            return
            
        with open(output_file, "w") as f:
            f.write("=== Dialogue Analysis Report ===\n\n")
            
            # 计算各模型平均差异
            f.write("Model Performance Summary:\n")
            f.write("--------------------------\n")
            for model_name, embeddings in analysis_results['generated_embeddings'].items():
                target = analysis_results['target_embeddings'][:len(embeddings)]
                differences = np.linalg.norm(embeddings - target, axis=1)
                
                f.write(f"\n{model_name}:\n")
                f.write(f"  - Average Difference: {np.mean(differences):.4f}\n")
                f.write(f"  - Max Difference: {np.max(differences):.4f}\n")
                f.write(f"  - Min Difference: {np.min(differences):.4f}\n")
                f.write(f"  - Valid Dialogues: {len(differences)}\n")
            
            # 最佳表现模型
            model_diffs = {
                model: np.mean(np.linalg.norm(embeddings - analysis_results['target_embeddings'][:len(embeddings)], axis=1))
                for model, embeddings in analysis_results['generated_embeddings'].items()
            }
            best_model = min(model_diffs, key=model_diffs.get)
            f.write(f"\nBest Performing Model: {best_model} (Avg Diff: {model_diffs[best_model]:.4f})\n")
            
        print(f"Report generated: {output_file}")
    
if __name__ == "__main__":
    start = time.time()
    
    print("\n⏳ Initializing analyzer...")
    analyzer = DialogueAnalyzer(model_dir="./model_cache")
    
    print("\n🚀 Analyzing dialogue data...")
    analysis_results = analyzer.analyze_dialogues("./merged_data.jsonl")
    
    print("\n📊 Generating visualizations...")
    analyzer.visualize_2d_differences(analysis_results)
    
    print("\n📝 Generating analysis report...")
    analyzer.generate_report(analysis_results)
    
    print(f"\n⏱️ Total time: {(time.time()-start)/60:.1f} minutes")
    print("\n✅ Analysis complete! Output files:")
    print("- 2d_diff_visualization.png")
    print("- analysis_report.txt")
