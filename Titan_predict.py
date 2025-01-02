import torch
import torch.nn.functional as F
import h5py
import joblib
import os
import yaml
import numpy as np
import pandas as pd


from transformers import PreTrainedModel, AutoModel
from datasets import load_from_disk
#from eval_linear_probe import train_and_evaluate_logistic_regression_with_val

'''
from .vision_transformer import build_vision_tower
from .text_transformer import build_text_tower
from .conch_v1_5 import build_conch
from .configuration_titan import TitanConfig
'''
from src.Titan.titan.utils import get_eval_metrics, TEMPLATES, bootstrap
from src.Titan.titan.eval_linear_probe import train_and_evaluate_logistic_regression_with_val
from src.Titan.create_patches import f_create_patches
from src.Titan.extract_features import f_extract_features

#joblib.dump(logistic_reg_final, 'logistic_reg_final.model')
#logistic_reg_final = joblib.load('logistic_reg_final.model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class C_Titan:
	疾病分类 = ('AASTR','ACC','AOAST','ASTR','BLCA','CCRCC','CESC','CHRCC','COAD','DDLS','DSTAD','ESCA','ESCC',
		'GBM','HCC','HGSOC','HNSC','IDC','ILC','LMS','LUAD','LUSC','MACR','MEL','MFH','MFS','NSGCT','OAST','ODG',
		'PAAD','PHC','PLEMESO','PRAD','PRCC','READ','SEM','SKCM','STAD','THFO','THPA','THYM','TSTAD','UCS','UEC','UM','USC')
	'''
	疾病分类 = ('间变性星形细胞瘤', '肾上腺皮质癌', '间变性寡星形细胞瘤;间变性混合性胶质瘤', '弥漫性星形细胞瘤;纤维型星形细胞瘤', 
		'尿路上皮癌', '透明细胞癌', '宫颈鳞状细胞癌', '嫌色细胞癌', '结肠腺癌', '去分化脂肪肉瘤', '弥漫性胃腺癌', '食管腺癌', '食管鳞状细胞癌',
		'多形性胶质母细胞瘤', '肝细胞癌', '高级别浆液性癌', '头颈鳞状细胞癌', '', '', '', '', '', '', '', '', '', '', '', '', 
		'', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '')
	'''
	def __init__(self, modelPath='./Models/Titan'):
		self.m_titanModle = AutoModel.from_pretrained(modelPath, trust_remote_code=True)
		#self.m_titanModle = AutoModel.from_pretrained('D:/Project/LaiDun_AI/Medical/TITAN-main/LaiDunMedical/Models/Titan/titan/', trust_remote_code=True)
		self.m_titanModle = self.m_titanModle.to(device)
		self.m_classificationModel = joblib.load(modelPath + '/logistic_reg_final.model')
		self.patch_size = np.int64(1024)
		self.svs文件 = ''
		self.图像区块h5文件 = ''
		self.区块特征h5文件 = ''
		pass

	def f_生成图像切片(self, source):
		self.patch_size = f_create_patches(source)
		

	def f_提取切片特征(self, data_dir):
		f_extract_features(data_dir, patch_size = self.patch_size)
		#f_extract_features(data_dir, self.m_titanModle)

	def f_生成图像特征(self, h5Path):
		file = h5py.File(h5Path, 'r')
		features = torch.from_numpy(file['features'][:])
		coords = torch.from_numpy(file['coords'][:])
		patch_size_lv0 = file['coords'].attrs['patch_size_level0']
		# load configs and prompts for TCGA-OT task
		with open('config/config_tcga-ot.yaml', 'r') as file:
		    task_config = yaml.load(file, Loader=yaml.FullLoader)
		class_prompts = task_config['prompts']
		target = task_config['target']
		label_dict = task_config['label_dict']
		# extract slide embedding
		with torch.autocast('cuda', torch.float16), torch.inference_mode():
			features = features.to(device)
			coords = coords.to(device)
			slide_embedding = self.m_titanModle.encode_slide_from_patch_features(features, coords, patch_size_lv0)
		#m_classificationModel.pre
		#print(slide_embedding)
		return slide_embedding.cpu().numpy()

	def f_打开文件(self):
		pass

	def f_2(self):
		pass

	def f_3(self):
		pass

	def f_4(self):
		pass

'''
#D:/Project/LaiDun_AI/Medical/TCGA_Data/84d08755-2f3a-4a27-acca-7b1e560e63fc

fPath = 'D:/Project/LaiDun_AI/Medical/TCGA_Data/a58e51d3-427a-4331-8819-2f9bfc1d0d84'
c_Titan = C_Titan()
#c_Titan.f_生成图像切片(fPath)
#c_Titan.f_提取切片特征(fPath)
l_slide_embedding = c_Titan.f_生成图像特征(fPath+'/feat/h5_files/TCGA-QM-A5NM-01Z-00-DX1.B4B9A4C5-AB04-4302-9C17-A7A88CEED873.h5')

l_分类结果 = c_Titan.m_classificationModel.predict(l_slide_embedding)
l_分类概率 = c_Titan.m_classificationModel.predict_proba(l_slide_embedding)[:, :]

# 通过enumerate获取索引和值，并构建字典
l_dict_分类概率 = {i: val for i, val in enumerate(l_分类概率[0])}

# 根据字典的值进行从大到小的排序
sorted_items = dict(sorted(l_dict_分类概率.items(), key=lambda item: item[1], reverse=True))
# 将排序后的项转换回字典类型
sorted_dict = dict(sorted_items)
for key, val in sorted_dict.items():
	print(C_Titan.疾病分类[int(key)] + ': ' + f"{val:.7f}")
'''