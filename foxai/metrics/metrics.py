import torch
import numpy as np
import matplotlib.pyplot as plt

from foxai.visualizer import _preprocess_img_and_attributes
from typing import List

def get_stepped_attrs(sorted_attrs:np.ndarray, sample_num:int)->np.ndarray:
        total_samples:int = len(sorted_attrs)
        required_step:int = total_samples//sample_num
        return sorted_attrs[::required_step]

def _metric_calculation(attrs:torch.tensor, transformed_img:torch.tensor, model:torch.nn.Module, chosen_class:int,steps_num=30, metric_type="DELETION"):
    if metric_type not in ["INSERTION", "DELETION"]:
        raise AttributeError(f"Metric type not in {['INSERTION', 'DELETION']}")
        
    attributes_matrix: np.ndarray = attrs.detach().cpu().numpy()
    transformed_img_np: np.ndarray = transformed_img.detach().cpu().numpy()
    
    preprocessed_attrs, _ = _preprocess_img_and_attributes(attributes_matrix=attributes_matrix, transformed_img_np=transformed_img_np, only_positive_attr=True)
    
    sorted_attrs:np.ndarray = np.flip(np.sort(np.unique(preprocessed_attrs)))
    stepped_attrs:np.ndarray = get_stepped_attrs(sorted_attrs, steps_num)
    
    importance_lst:list = []
    for val in stepped_attrs:
        
        attributes_map:np.ndarray = np.expand_dims(np.where(preprocessed_attrs <= val, 1, 0), axis=-1)
        attributes_map = attributes_map.repeat(3, axis=-1)
        attributes_map = torch.from_numpy(attributes_map).permute(2,1,0)

        attributes_map_inv:np.ndarray = np.expand_dims(np.where(preprocessed_attrs <= val, 0, 1), axis=-1)
        attributes_map_inv = attributes_map_inv.repeat(3, axis=-1)
        attributes_map_inv = torch.from_numpy(attributes_map_inv).permute(2,1,0)

        mean_img:torch.tensor = torch.zeros(transformed_img.shape)
        mean_img[:] = transformed_img.mean()
        
        if metric_type == "DELETION":
            perturbed_img:torch.tensor = transformed_img * attributes_map + mean_img*attributes_map_inv
        else:
             perturbed_img:torch.tensor = mean_img * attributes_map + transformed_img*attributes_map_inv
                
        output = model(perturbed_img.unsqueeze(dim=0))
        softmax_output:torch.tensor = torch.nn.functional.softmax(output)[0]
        importance_lst.append(softmax_output[chosen_class].detach().numpy())

    metric:np.ndarray = np.round(np.trapz(importance_lst)/len(importance_lst), 4)
    
    return metric, importance_lst  

def deletion(attrs:torch.tensor, transformed_img:torch.tensor, model:torch.nn.Module, chosen_class:int):
    return _metric_calculation(attrs, transformed_img, model, chosen_class, metric_type="DELETION")

def insertion(attrs:torch.tensor, transformed_img, model:torch.nn.Module, chosen_class):
     return _metric_calculation(attrs, transformed_img, model, chosen_class, metric_type="INSERTION")
    
def visualize_metric(importance_lst:List, metric_result:float, metric_type:str="Deletion"):
    plt.ylim((0,1))
    plt.xlim((0,len(importance_lst)))
    plt.plot(np.arange(len(importance_lst)), importance_lst)
    plt.title(f"{metric_type}: {metric_result}")
    plt.show()
    
                             