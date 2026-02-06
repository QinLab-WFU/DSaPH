import os
import scipy.io as scio
import torch
import numpy as np


# def Save_mat(self , query_img , retrieval_img , mode_name="i2t"):
#     save_dir = os.path.join(self.args.save_dir , "PR_cruve")
#     os.makedirs(save_dir,exist_ok=True)
#
#     query_img = query_img.cpu().detach().numpy()
#     retrieval_img = retrieval_img.cpu().detach().numpy()
#
#     query_label = self.query_labels.numpy()
#     retrieval_label = self.retrieval_labels.numpy()
#
#     result_dict = {
#         'q_img' : query_img ,
#         'r_img' : retrieval_img ,
#         'q_l' : query_label ,
#         'r_l' : retrieval_label
#     }
#
#     scio.savemat(os.path.join(save_dir , str(self.args.ouput_dim)
#                  + "-ours-" + self.args.datasets + "-" , + mode_name + ".mat"),result_dict)
def Save_mat(epoch: object, output_dim: object, datasets: object, query_labels: object, retrieval_labels: object, query_img: object, retrieval_img: object,
             save_dir: object = '.',
             mode_name: object = "DSH",
             mAP: object = 0) -> object:
    '''
    save_dir: 保存文件的目录路径
    output_dim: 输出维度
    datasets: 数据集名称
    query_labels: 查询图像的标签信息（numpy数组）
    retrieval_labels: 检索图像的标签信息（numpy数组）
    query_img: 查询图像的数据（numpy数组）
    retrieval_img: 检索图像的数据（numpy数组）
    mode_name: 模型的名称
    '''
    save_dir = os.path.join(save_dir , f'Hashcode_{datasets}_{output_dim}_{mode_name}')
    os.makedirs(save_dir,exist_ok=True)

    query_img = query_img.cpu().detach().numpy()
    retrieval_img = retrieval_img.cpu().detach().numpy()
    print(query_img.shape)
    print(retrieval_labels.shape)
    # query_label = query_labels.cpu().numpy()
    # retrieval_label = retrieval_labels.cpu().numpy()

    result_dict = {
        'q_img' : query_img ,
        'r_img' : retrieval_img ,
        'q_l' : query_labels ,
        'r_l' : retrieval_labels
    }

    filename = os.path.join(save_dir, f"{mAP}_{output_dim}-{epoch}-{datasets}-{mode_name}.mat")
    scio.savemat(filename, result_dict)

