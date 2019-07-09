# knowledge-driven-dialogue-2019-lic
2019语言与智能技术竞赛[知识驱动对话](http://lic2019.ccf.org.cn/talk) B榜第5名方案<br>
由于线上部署对时间有要求，最终提交人工评估的版本删掉了一些全局主题特征，导致模型结果有所下降，最终人工评估第9名。A榜第四 B榜第五
## Overview
For building a proactive dialogue chatbot, we used a so-called generation-reranking method. First, the generative models(Multi-Seq2Seq) produce some candidate replies. Next, the re-ranking model is responsible for performing query-answer matching, to choice a reply as informative as possible over the produced candidates.  A detailed paper to describle our solution is now avaliable at https://arxiv.org/pdf/1907.03590.pdf, please check.
### Data Augmentation
We used four data augmentation techniques, Entity Generalization,Knowledge Selection,Switch,Conversation Extraction to construct multiple different dataset for training Seq2Seq models. One can use the scripts Seq2Seq/preclean_*.py to with slight modification of parameters to get 6 datasets.
### Seq2Seq Model
For ensemble purpose we choose different encoders and decoders, i.e. LSTM cells and the Transformer, for model diversity. This part is implemented based on the [Open-NMT](https://github.com/OpenNMT/OpenNMT-py) framework. <br>
#### Training
- python preprocess.py
- python train.py
#### Testing
- python translate.py
All the config file of training & testing can be easily modified in the config/*.yml <br>
In total, we trained 27 Seq2Seq model for ensemble.
### Answer rank
We used a GBDT regressor for ranking. One may arugue that Why not use a neural network, such as BERT for ranking. Actually We tried, but it doesn't work well.
#### Creating ranking dataset
python create_gbdt_dataset.py
#### Feature extraction
python feature_util_multiprocess.py <br>
The feature extractions reference the [Kaggle_HomeDepot](https://github.com/ChenglongChen/Kaggle_HomeDepot) by ChenglongChen 
### Checkpoints
It might take some extra time to upload the checkpoints because they are rather large in size.
