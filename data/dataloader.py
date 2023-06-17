import sys
sys.path.append('..')
from data.gc_dataloader import dataloader_gc
from data.lp_dataloader import dataloader_lp



def dataloader(task_type, model_name, dataset_name, collection_name, device="cpu"):
    if task_type not in ['nc','lp','gc']:
        print("Task type should in ['nc','lp','gc']")
        raise NotImplementedError
    if task_type=='nc':
        pass
    elif task_type=="lp":
        return dataloader_lp(model_name, dataset_name, collection_name, device)
    elif task_type=="gc":
        return dataloader_gc(model_name, dataset_name, collection_name, device)
    else:
        raise NotImplementedError


