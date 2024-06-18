# Emotion Recognition on Accented Speech using Domain Adaptation

Transfer learning has become an important tool to use in various domains of Artificial Intelligence to reproduce results on problems with less or inconsistent data. Domain adaptation is a subset of transfer learning, that deals with different data distributions within the same task. The aim of this paper is to transfer a Speech Emotion Recognition (SER) model from standard English to Singaporean-accented English. We report on baselines and transfer experiments for the speech emotion recognition task on the MSP-Podcast and NSC speech datasets. These results set a foundation for future experiments to better understand how to use domain adaptation to increase performance in accented speech domains. 


This GitHub Repo is structured into three folders: `dataloader`, `inference`, and `training`. Each of which contain Python notebook scripts which can be run to reproduce our experiments in speech emotion recognition, as outlined in our final report.

- The dataloader folder contains a notebook for creating custom dataloaders to be used for training and inference of the Emotion Classification using audio.
- The training folder contains 4 notebooks as described:
  - Wav2vec2_Emotion_Recognition.ipynb: This notebook contains the code for training a model to perform speech emotion classification using the Wav2vec2 model.
  - HuBERT_for_Emotion_Recognition.ipynb: This notebook contains the code for training a model to perform speech emotion classification using the HuBERT model.
  - XLSR_for_Emotion_Recognition.ipynb: This notebook contains the code for training a model to perform speech emotion classification using the XLSR model.
  - Domain_Adaptation_for_Emotion_Transfer_on_Accented_Speech.ipynb: This notebook contains the code to first create the baseline model, implement a Domain Adaptation model (reference: https://sites.skoltech.ru/compvision/projects/grl/), and train the domain adapted model.
- The inference folder contains 3 notebooks as described:
  - wav2vec2-inference.ipynb: This notebook contains the code for performing inference for emotion classification using Wav2vec2 model.
  - hubert-inference.ipynb:  This notebook contains the code for performing inference for emotion classification using HuBERT model.
  - xlsr-inference.ipynb:  This notebook contains the code for performing inference for emotion classification using XLSR model.

## Dataset

For our experiments, we train and collect evaluation metrics on the MSP-Podcast database (Lotfian and Busso, 2019) and National Speech Corpus (Koh et al., 2019).
Some more information on the datasets:


<div style="text-align:center">
  <img width="768" alt="Screen Shot 2024-06-18 at 12 04 36 AM" src="https://github.com/faizankhan29/Emotion-Recognition-on-Accented-Speech-using-Domain-Adaptation/assets/10673214/4410cb55-d2f2-4ce0-a426-b03ca9f6049e">
</div>

## Baseline Architectures

For our baseline experiments, we adapted three state-of-the-art models in speech recognition for SER as shown in Figure 1. In each approach, we leverage the state-of-the-art pretrained architectures as feature extractors and finetune classification with a speech emotion recognition head.

We train each baseline on the subset of English-accented MSP-Podcast train data for 10 epochs and evaluate on the English-accented MSP-Podcast validation and Singaporean-English accented NSC emotion test set.

 <div style="text-align:center">
<img width="776" alt="Screen Shot 2024-06-18 at 12 07 33 AM" src="https://github.com/faizankhan29/Emotion-Recognition-on-Accented-Speech-using-Domain-Adaptation/assets/10673214/d09cc7b7-d1c0-4ef7-bf54-6655cd7ae367">
</div>

## Domain Adaptation: Domain Adversarial Neural Network (DANN)

Domain adaptation specializes in improving model performance in cases where there is a difference in distributions amongst different data sets. For instance, this difference could look like a data shift, a shift in the variance of the variable or a shift in the mean of the variable due to various factors such as the training and test data sets originating from varying sources Farahani et al. (2020).
Domain Adversarial Neural Networks (DANN) is an approach within domain adaptation that has labeled data at the source and unlabelled data at the target Ganin et al. (2015). During the training of a DANN, certain features appear that do not differentiate as we move between domains. As shown in Figure 2 the Domain Adversarial Neural Network architecture produces features for label and domain classifier and then predicts the class label and domains.

<div style="text-align:center">
<img width="754" alt="Screen Shot 2024-06-18 at 12 11 19 AM" src="https://github.com/faizankhan29/Emotion-Recognition-on-Accented-Speech-using-Domain-Adaptation/assets/10673214/3b330438-e8ff-4c95-84c4-4b5c76a999a9">
</div>

## Results

### Transfer Learning Results

We report hyperparameter fine-tuned baseline results on MSP-Podcast across all three proposed models: Wav2Vec 2.0, HuBERT, and XLSR. For each model, we report accuracy, weighted F1, and macro on the MSP validation and NSC test splits. In Table 4, we outline a comparison of each model. We train over 10 epochs. Each model took around 10-15 training hours for 10 epochs on a 40GB NVIDIA GTX A6000. Performance reported is from the best-performing model over 10 epochs.


<img width="768" alt="Screen Shot 2024-06-18 at 12 13 24 AM" src="https://github.com/faizankhan29/Emotion-Recognition-on-Accented-Speech-using-Domain-Adaptation/assets/10673214/1bc8fc91-c071-419c-b2dd-c3786b7685a9">


### DANN Results

In the following table, we report results from our DANN experiments. We train the architecture for 10 epochs on a 40 GB A100 Google Colab GPU. We report results on MSP and NSC test data from the highest performing checkpoint over training.


<img width="748" alt="Screen Shot 2024-06-18 at 12 15 00 AM" src="https://github.com/faizankhan29/Emotion-Recognition-on-Accented-Speech-using-Domain-Adaptation/assets/10673214/dd365377-4358-4823-a6da-f326fc0a3521">


## Future Work and Limitations

In conclusion, we report our transfer learning and domain adaptation results. **We find that the DANN implementation does not improve against transfer learning baselines.** We hypothesize that the DANN architecture did not surpass baselines due to insufficient distributions of train data. In particular, we note that our NSC emotion dataset was small and training with more NSC emotion labels would likely improve transfer more effectively as the model would better train the domain classifier to understand the distribution shift. As such, in future work, we hope to have more even batches between MSP and NSC train data.
Despite the success in the transfer learning baselines, we highlight limitations in imbalance of our datasets, which likely contribute a skewed over exaggerated performance on our accented test split. To address this issue, we hope to evaluate these models on a more balanced and larger speech emotion test dataset. Moreover, if we accept imbalance of distribution as a natural aspect of emotion recognition in general, future work may also focus on techniques to address dataset imbalance.


