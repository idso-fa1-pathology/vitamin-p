"""
Test evaluation with per-dataset breakdown and PQ metrics
"""

import torch
from tqdm import tqdm
from collections import defaultdict

# Change from .metrics to metrics (parent-level import)
from metrics import (
    dice_coefficient,
    iou_score,
    get_fast_pq,
    get_fast_aji,
    get_fast_aji_plus
)

from .utils import SimplePreprocessing, prepare_he_input, prepare_mif_input

class TestEvaluator:
    """
    Comprehensive test set evaluation with PQ metrics and per-dataset breakdown
    
    Args:
        model: VitaminP model instance
        device: Device ('cuda' or 'cpu')
        model_type: 'Dual', 'Flex', 'BaselineHE', 'BaselineMIF'
    """
    def __init__(self, model, device, model_type):
        self.model = model
        self.device = device
        self.model_type = model_type
        self.preprocessor = SimplePreprocessing()
    
    @torch.no_grad()
    def evaluate(self, test_loader):
        """
        Evaluate on test set with per-dataset breakdown
        
        Returns:
            Dictionary with overall metrics and per-dataset breakdown
        """
        if test_loader is None:
            return None
        
        self.model.eval()
        
        # Overall metrics
        overall_metrics = self._init_metrics()
        overall_counts = {'he': 0, 'he_cell': 0, 'mif': 0, 'total': 0}
        
        # Per-dataset metrics
        dataset_metrics = defaultdict(lambda: self._init_metrics())
        dataset_counts = defaultdict(lambda: {'he': 0, 'he_cell': 0, 'mif': 0, 'total': 0})
        
        pbar = tqdm(test_loader, desc='Test Evaluation', ncols=100)
        
        for batch in pbar:
            sources = batch['dataset_source']
            batch_size = len(sources)
            
            # Route to appropriate evaluation based on model type
            if self.model_type == 'Flex':
                batch_results = self._evaluate_flex_batch(batch)
            elif self.model_type == 'BaselineHE':
                batch_results = self._evaluate_baseline_he_batch(batch)
            elif self.model_type == 'BaselineMIF':
                batch_results = self._evaluate_baseline_mif_batch(batch)
            else:  # Dual/Syn
                batch_results = self._evaluate_dual_batch(batch)
            
            # Aggregate results
            for i, src in enumerate(sources):
                sample_metrics = batch_results[i]
                
                # Update overall
                self._add_metrics(overall_metrics, sample_metrics)
                overall_counts['total'] += 1
                if sample_metrics.get('he_nuclei_dice') is not None:
                    overall_counts['he'] += 1
                if sample_metrics.get('he_cell_dice') is not None:
                    overall_counts['he_cell'] += 1
                if sample_metrics.get('mif_nuclei_dice') is not None:
                    overall_counts['mif'] += 1
                
                # Update per-dataset
                self._add_metrics(dataset_metrics[src], sample_metrics)
                dataset_counts[src]['total'] += 1
                if sample_metrics.get('he_nuclei_dice') is not None:
                    dataset_counts[src]['he'] += 1
                if sample_metrics.get('he_cell_dice') is not None:
                    dataset_counts[src]['he_cell'] += 1
                if sample_metrics.get('mif_nuclei_dice') is not None:
                    dataset_counts[src]['mif'] += 1
        
        # Average metrics
        overall_avg = self._average_metrics(overall_metrics, overall_counts)
        
        dataset_avg = {}
        for src in dataset_metrics.keys():
            dataset_avg[src] = self._average_metrics(dataset_metrics[src], dataset_counts[src])
        
        return {
            'overall': overall_avg,
            'per_dataset': dataset_avg
        }
    
    def _init_metrics(self):
        """Initialize metric dictionary"""
        return {
            'he_nuclei_dice': 0, 'he_nuclei_iou': 0, 'he_nuclei_pq': 0, 'he_nuclei_aji': 0,
            'he_cell_dice': 0, 'he_cell_iou': 0, 'he_cell_pq': 0, 'he_cell_aji': 0,
            'mif_nuclei_dice': 0, 'mif_nuclei_iou': 0, 'mif_nuclei_pq': 0, 'mif_nuclei_aji': 0,
            'mif_cell_dice': 0, 'mif_cell_iou': 0, 'mif_cell_pq': 0, 'mif_cell_aji': 0,
        }
    
    def _add_metrics(self, target, source):
        """Add source metrics to target (accumulate)"""
        for key, val in source.items():
            if val is not None:
                target[key] += val
    
    def _average_metrics(self, metrics, counts):
        """Average accumulated metrics"""
        result = {}
        
        # HE metrics
        if counts['he'] > 0:
            result['he_nuclei_dice'] = metrics['he_nuclei_dice'] / counts['he']
            result['he_nuclei_iou'] = metrics['he_nuclei_iou'] / counts['he']
            result['he_nuclei_pq'] = metrics['he_nuclei_pq'] / counts['he']
            result['he_nuclei_aji'] = metrics['he_nuclei_aji'] / counts['he']
        
        if counts['he_cell'] > 0:
            result['he_cell_dice'] = metrics['he_cell_dice'] / counts['he_cell']
            result['he_cell_iou'] = metrics['he_cell_iou'] / counts['he_cell']
            result['he_cell_pq'] = metrics['he_cell_pq'] / counts['he_cell']
            result['he_cell_aji'] = metrics['he_cell_aji'] / counts['he_cell']
        
        # MIF metrics
        if counts['mif'] > 0:
            result['mif_nuclei_dice'] = metrics['mif_nuclei_dice'] / counts['mif']
            result['mif_nuclei_iou'] = metrics['mif_nuclei_iou'] / counts['mif']
            result['mif_nuclei_pq'] = metrics['mif_nuclei_pq'] / counts['mif']
            result['mif_nuclei_aji'] = metrics['mif_nuclei_aji'] / counts['mif']
            
            result['mif_cell_dice'] = metrics['mif_cell_dice'] / counts['mif']
            result['mif_cell_iou'] = metrics['mif_cell_iou'] / counts['mif']
            result['mif_cell_pq'] = metrics['mif_cell_pq'] / counts['mif']
            result['mif_cell_aji'] = metrics['mif_cell_aji'] / counts['mif']
        
        # Overall average
        all_dice = [v for k, v in result.items() if 'dice' in k]
        result['dice_avg'] = sum(all_dice) / len(all_dice) if all_dice else 0
        
        all_pq = [v for k, v in result.items() if '_pq' in k]
        result['pq_avg'] = sum(all_pq) / len(all_pq) if all_pq else 0
        
        return result
    
    def _evaluate_flex_batch(self, batch):
        """Evaluate Flex model batch"""
        sources = batch['dataset_source']
        batch_size = len(sources)
        results = []
        
        for i in range(batch_size):
            src = sources[i]
            metrics = {}
            
            # Determine modality
            if src == 'tissuenet':
                use_mif = True
            elif src in ['pannuke', 'lizard', 'monuseg', 'tnbc', 'nuinsseg', 'cryonuseg', 'bc', 'consep', 'monusac', 'kumar', 'cpm17']:
                use_mif = False
            else:
                use_mif = torch.rand(1).item() < 0.5
            
            if use_mif:
                img = prepare_mif_input(batch['mif_image'][i:i+1].to(self.device))
                img = self.preprocessor.percentile_normalize(img)
                outputs = self.model(img)
                
                # ✅ FIX: Use index 0 because outputs is single sample
                metrics.update(self._compute_sample_metrics(
                    outputs, batch, i, 'mif_nuclei', 'mif_nuclei', pred_idx=0
                ))
                metrics.update(self._compute_sample_metrics(
                    outputs, batch, i, 'mif_cell', 'mif_cell', pred_idx=0
                ))
            else:
                img = prepare_he_input(batch['he_image'][i:i+1].to(self.device))
                img = self.preprocessor.percentile_normalize(img)
                outputs = self.model(img)
                
                # ✅ FIX: Use index 0 because outputs is single sample
                metrics.update(self._compute_sample_metrics(
                    outputs, batch, i, 'he_nuclei', 'he_nuclei', pred_idx=0
                ))
                
                # HE cell (skip for PanNuke/Lizard)
                if src not in ['pannuke', 'lizard', 'monuseg', 'tnbc', 'nuinsseg', 'cryonuseg', 'bc', 'consep', 'monusac', 'kumar', 'cpm17']:
                    metrics.update(self._compute_sample_metrics(
                        outputs, batch, i, 'he_cell', 'he_cell', pred_idx=0
                    ))
            
            results.append(metrics)
        
        return results

    def _evaluate_baseline_he_batch(self, batch):
        """Evaluate BaselineHE batch"""
        sources = batch['dataset_source']
        he_img = batch['he_image'].to(self.device)
        he_img = self.preprocessor.percentile_normalize(he_img)
        outputs = self.model(he_img)
        
        batch_size = he_img.shape[0]
        results = []
        
        for i in range(batch_size):
            src = sources[i]
            metrics = {}
            
            # HE nuclei (always)
            metrics.update(self._compute_sample_metrics(
                outputs, batch, i, 'he_nuclei', 'he_nuclei'
            ))
            
            # HE cell (skip for PanNuke/Lizard)
            if src not in ['pannuke', 'lizard', 'monuseg', 'tnbc', 'nuinsseg', 'cryonuseg', 'bc', 'consep', 'monusac', 'kumar', 'cpm17']:
                metrics.update(self._compute_sample_metrics(
                    outputs, batch, i, 'he_cell', 'he_cell'
                ))
            
            results.append(metrics)
        
        return results
    
    def _evaluate_baseline_mif_batch(self, batch):
        """Evaluate BaselineMIF batch"""
        mif_img = batch['mif_image'].to(self.device)
        mif_img = self.preprocessor.percentile_normalize(mif_img)
        outputs = self.model(mif_img)
        
        batch_size = mif_img.shape[0]
        results = []
        
        for i in range(batch_size):
            metrics = {}
            metrics.update(self._compute_sample_metrics(outputs, batch, i, 'mif_nuclei', 'mif_nuclei'))
            metrics.update(self._compute_sample_metrics(outputs, batch, i, 'mif_cell', 'mif_cell'))
            results.append(metrics)
        
        return results
    
    def _evaluate_dual_batch(self, batch):
        """Evaluate Dual/Syn batch"""
        he_img = batch['he_image'].to(self.device)
        mif_img = batch['mif_image'].to(self.device)
        
        he_img = self.preprocessor.percentile_normalize(he_img)
        mif_img = self.preprocessor.percentile_normalize(mif_img)
        
        outputs = self.model(he_img, mif_img)
        
        batch_size = he_img.shape[0]
        results = []
        
        for i in range(batch_size):
            metrics = {}
            metrics.update(self._compute_sample_metrics(outputs, batch, i, 'he_nuclei', 'he_nuclei'))
            metrics.update(self._compute_sample_metrics(outputs, batch, i, 'he_cell', 'he_cell'))
            metrics.update(self._compute_sample_metrics(outputs, batch, i, 'mif_nuclei', 'mif_nuclei'))
            metrics.update(self._compute_sample_metrics(outputs, batch, i, 'mif_cell', 'mif_cell'))
            results.append(metrics)
        
        return results
    
    def _compute_sample_metrics(self, outputs, batch, batch_idx, output_key, gt_key, pred_idx=0):
        """Compute all metrics for one sample and one task
        
        Args:
            outputs: Model outputs
            batch: Batch data
            batch_idx: Index in the batch for ground truth
            output_key: Key for output (e.g., 'he_nuclei')
            gt_key: Key for ground truth (e.g., 'he_nuclei')
            pred_idx: Index in outputs (usually 0 for single-sample forward pass)
        """
        try:
            # ✅ Use pred_idx for outputs, batch_idx for ground truth
            pred_seg = outputs[f'{output_key}_seg'][pred_idx].cpu()
            pred_hv = outputs[f'{output_key}_hv'][pred_idx].cpu()
            
            gt_seg = batch[f'{gt_key}_mask'][batch_idx].unsqueeze(0).cpu()
            gt_instance = batch[f'{gt_key}_instance'][batch_idx].cpu().numpy()
            
            # ✅ Generate instance map using HV post-processing
            from postprocessing import process_model_outputs
            
            # Extract h and v maps (HV tensor is [2, H, W])
            h_map = pred_hv[0].numpy()  # First channel = horizontal
            v_map = pred_hv[1].numpy()  # Second channel = vertical
            seg_pred = pred_seg.squeeze().numpy()
            
            # Apply HV post-processing to get instance map
            pred_instance, inst_info, num_instances = process_model_outputs(
                seg_pred=seg_pred,
                h_map=h_map,
                v_map=v_map,
                magnification=40,
                binary_threshold=0.5
                # ← Removed use_gpu parameter
            )
            
            # Binary metrics (need to add batch dimension back)
            dice = dice_coefficient(torch.from_numpy(seg_pred)[None, None, ...], gt_seg)
            iou = iou_score(torch.from_numpy(seg_pred)[None, None, ...], gt_seg)
            
            # Instance metrics
            pq, _, _ = get_fast_pq(gt_instance, pred_instance)
            aji = get_fast_aji(gt_instance, pred_instance)
            
            return {
                f'{output_key}_dice': dice,
                f'{output_key}_iou': iou,
                f'{output_key}_pq': pq,
                f'{output_key}_aji': aji
            }
        except Exception as e:
            import traceback
            print(f"⚠️  Metric computation failed for {output_key}: {e}")
            traceback.print_exc()  # ← Add this to see full error
            return {
                f'{output_key}_dice': None,
                f'{output_key}_iou': None,
                f'{output_key}_pq': None,
                f'{output_key}_aji': None
            }