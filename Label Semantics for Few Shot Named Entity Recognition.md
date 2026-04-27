# Label Semantics for Few Shot Named Entity Recognition

Jie Ma $^{1}$ Miguel Ballesteros $^{1}$ Srikanth Doss $^{1}$ Rishita Anubhai $^{1}$ Sunil Mallya $^{1*}$ Yaser Al-Onaizan $^{1*}$ Dan Roth $^{1,2}$ ${}^{1}\mathrm{AWS}$ AI Labs 

$^{2}$ Computer and Information Science, University of Pennsylvania 

{jieman, ballemig, srikad, ranubhai, drot}@amazon.com mallya16@gmail.com, onaizan2000@yahoo.com 

# Abstract

We study the problem of few shot learning for named entity recognition. Specifically, we leverage the semantic information in the names of the labels as a way of giving the model additional signal and enriched priors. We propose a neural architecture that consists of two BERT encoders, one to encode the document and its tokens and another one to encode each of the labels in natural language format. Our model learns to match the representations of named entities computed by the first encoder with label representations computed by the second encoder. The label semantics signal is shown to support improved state-of-the-art results in multiple few-shot NER benchmarks and on-par performance in standard benchmarks. Our model is especially effective in low resource settings. 

# 1 Introduction

Named entity recognition (NER) seeks to locate named entity spans in unstructured text and classify them into pre-defined categories such as PERSON, LOCATION and ORGANIZATION (Tjong Kim Sang and De Meulder, 2003a). As a fundamental natural language understanding task, NER often serves as an upstream component for more complex tasks such as question answering (Mollá et al., 2006), relation extraction (Chan and Roth, 2011) and coreference resolution (Clark and Manning, 2015). However, building an accurate NER system has traditionally required large amounts of high quality annotated in-domain data (Lison et al., 2020; Chen et al., 2020). This usually involves well defined annotation guidelines and training of annotators, which requires rich domain knowledge and can be prohibitively expensive (Huang et al., 2020). 

Few shot learning (FSL) (Vinyals et al., 2017; Finn et al., 2017; Snell et al., 2017) aims at performing a task using only very few annotated examples (i.e. support set). 

Similarity-based methods, such as prototypical networks, are extensively studied and show great success for FSL (Vinyals et al., 2017; Snell et al., 2017; Yu et al., 2018a; Hou et al., 2020). The core idea is to classify input examples from a new domain based on their similarities with representations of each class in the support set. These methods do not utilize the semantics of label names and usually represent labels by directly averaging the embedding of support set examples, oversimplifying the learning of label representations. The main premise of our work is that label names carry meaning that our models can induce from data; the labels are themselves words that appear in text in various contexts and are thus semantically related to other words that appear in text, and this relatedness can be leveraged. For example, the representation of "Lionel Messi" is more similar to that of PERSON than to the representations of LOCATION or DATE when similar priors are used for labels and words or phrases. 

In this work, we propose a neural architecture that uses two separate BERT-based encoders (Devlin et al., 2019) to leverage semantics of label names for NER. One encoder (a) is used to encode the document and its words while the other encoder (b) is used to encode label names (e.g. PERSON, LOCATION etc.). The model is trained to match word representations from encoder (a) with label representations from encoder (b), and assign a label for each word by maximizing the 

similarity. We also experiment by replacing the BERT label encoder with GloVe embeddings (Pennington et al., 2014) as a simplified architecture. 

We report experimental results in multiple NER datasets from different domains. We summarize our contribution as follows: 

- We propose a simple and effective model architecture that leverages label semantics for NER. 

- We show that the proposed model is particularly effective in low resource settings and gives on-par results with the state-of-the-art models in high resource settings. 

- We achieve a new state-of-the-art in multiple few shot NER benchmarks. Specifically, our model outperforms prior work by 1.2 to 6.6 F1 points on CoNLL'03, WNUT'17, JNLPBA, NCBI-disease and I2B2'14 datasets on various few shot shots settings (§3.6). 

- We show that the proposed model is robust to variations of label names and that it is able to differentiate semantically similar labels. 

# 2 Model

We present our NER model. As shown in Figure 1, it consists of two BERT-based encoders where one encoder is used to encode the document and its tokens and the other to encode labels. We formalize the differences between datasets used in our experimentation (§2.1), then present how two BERT-based encoders (and the modification with GloVe-based encoder for labels) are used to leverage semantics in labels for NER (§2.2). Finally we discuss the training procedure (§2.3) and how labels are represented (§2.4). 

# 2.1 Source and Target Datasets

For few shot NER, we use a setup similar to meta-learning. We first train our models on source datasets $\{\mathcal{D}_1^S,\mathcal{D}_2^S,\ldots \}$ , then evaluate the model on unseen few shot target datasets $\{\mathcal{D}_1^T,\mathcal{D}_2^T,\ldots \}$ with or without finetuning. Each target dataset only contains a few examples and a different taxonomy of labels compared to the source datasets. 

# 2.2 Architecture

We use two BERT-based encoders as shown in Figure 1: a BERT document encoder and a BERT label encoder (we also experiment with GloVe embeddings as label encoder, described in §3.5). Like the traditional NER models (Carreras et al., 2003; Collobert et al., 2011; Lample et al., 2016, inter alia), we predict the label of each token with BIO scheme. For each token we get an embedding $e$ from the first BERT document encoder. For the unique set of labels $\mathcal{L}_D$ associated with dataset $D$ , we apply three steps to get the representations: First, we manually convert the label names to their natural language forms, e.g. "PER" to "person", "ORG" to "organization" etc. Second, we convert each of the label names to BIO scheme, in the form of natural language, e.g. "person" to "begin person" or "inside person". Finally, we use the second BERT label encoder to embed each of the labels in natural language BIO scheme. We compute the BERT [CLS] token embedding as the representation for the corresponding label. We form a label vector $\pmb{b}$ of all label embeddings $b_i$ for all $i$ in $\{1, 2, \dots, 2 \times N_L - 1\}^3$ . The label encoder acts like a lookup table for label embeddings. Finally, to find the most appropriate label for this token, we use: 

$$
y = \underset {i} {\arg \max } \operatorname {s o f t m a x} (e \cdot \boldsymbol {b})
$$

# 2.3 Training

Comparing with prior work on neural architectures for NER, our model does not require a new randomly initialized top layer classifier for a new dataset with new unseen label names. Instead, we generate label representations from the BERT label encoder. We hypothesize that this is beneficial because it prevents the model from forgetting priors since no parameters are dropped or randomly initialized for different datasets. 

We propose a simple two stage training procedure. In the first stage, we pre-finetune our model on the mix of all source datasets (which usually have different label set taxonomies), then we fine 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-27/193d6cc2-1574-4554-b7cb-92d5b465075b/36d738f21eca69b3d3da5e2a7c64307472983f410222e153c134ae6a7a77524c.jpg)



Figure 1: The architecture of our NER model. The diagram shows how representation of labels and tokens are produced, and how we use them to calculate final model prediction. The top part of the figure shows how labels are encoded; the bottom part of the figure shows how sentence are encoded.


tune the trained model on the target dataset. This process is also known as pre-finetuning (Aghajanyan et al., 2021) and finetuning. For scenarios where no source datasets are available, we simply skip the first stage. During model training time, both encoders are updated for every iteration at both stages, which helps to align the token embedding space and the label embedding space. 

During inference time, the learned label encoder is only required to produce label representations once. This is because the label representations may be cached and the label encoder is no longer needed to recompute representations. Our model is therefore not introducing additional memory overhead (since label encoder is removed) or latency overhead (since label representation is cached). 

# 2.4 Label Representation

Given that our label encoder is based on BERT and contains the priors from pretraining, our architecture allows any textual form as input for the generation of label representations. In order to make our results comparable with previous studies, we use only the natural language form of label names for our primary results. We discuss more label representations in Appendix E. 

# 3 Experiments

We evaluate our model and we compare it against existing few shot methods in two scenarios: high 

resource and low resource (few shot). In both cases, we assume there is a source dataset (which may be a set) with abundant data, and our goal is to maximize model performance on unseen target datasets which follow different taxonomies from the source dataset. 

# 3.1 Datasets

We perform experiments on 6 NER datasets from 5 different domains: OntoNotes 5.0 (Weischedel et al., 2013) (Mixed), CoNLL-2003 (Tjong Kim Sang and De Meulder, 2003a) (News), WNUT-2017 (Derczynski et al., 2017) (Social), JNLPBA (Collier and Kim, 2004) (Biology), NCBI-disease (Dogan et al., 2014) (Biology) and I2B2-2014 (Stubbs and Uzuner, 2015) (Medical). In all our experiments and following the definition in 2.1, we treat OntoNotes as the source dataset and all other as target datasets. $^{4}$ 

# 3.2 Settings and Evaluation

In this Section, we present the different experiments, and how do we carry out the evaluation. 

High Resource: Given a target dataset, we simply take all available data and evaluate on the standard held-out test set. 

<table><tr><td colspan="2"></td><td>1 Shot</td><td>5 Shot</td><td>20 Shot</td><td>50 Shot</td><td>Full Dataset</td></tr><tr><td rowspan="7">CoNLL-2003</td><td>TransferBERT</td><td>44.8 ±15.0</td><td>66.9 ±6.7</td><td>77.5 ±1.2</td><td>82.0 ±1.1</td><td>91.3 ±0.2</td></tr><tr><td>Prototypical Network</td><td>7.5 ±2.6</td><td>11.5 ±5.6</td><td>18.6 ±7.5</td><td>16.3 ±2.7</td><td>N/A</td></tr><tr><td>WPN-CRF</td><td>56.26 ±9.1</td><td>67.7 ±4.4</td><td>67.4 ±2.0</td><td>69.0 ±1.7</td><td>N/A</td></tr><tr><td>Struct NN shot</td><td>63.7 ±3.7</td><td>70.0 ±3.0</td><td>73.1 ±1.9</td><td>75.7 ±1.8</td><td>N/A</td></tr><tr><td>TANL</td><td>54.7 ±9.4</td><td>65.6 ±3.8</td><td>71.0 ±2.4</td><td>74.4 ±1.9</td><td>91.7 ±0.4</td></tr><tr><td>Our model - GloVe</td><td>63.1 ±6.9</td><td>73.5 ±2.4</td><td>78.3 ±1.1</td><td>82.0 ±1.5</td><td>91.6 ±0.2</td></tr><tr><td>Our model - BERT</td><td>68.4 ±6.7</td><td>76.6 ±2.1</td><td>79.7 ±1.1</td><td>83.1 ±1.2</td><td>91.5 ±0.2</td></tr><tr><td rowspan="7">WNUT-2017</td><td>TransferBERT</td><td>27.6 ±6.8</td><td>35.2 ±3.4</td><td>40.9 ±1.6</td><td>42.5 ±1.2</td><td>44.0 ±0.2</td></tr><tr><td>Prototypical Network</td><td>1.7 ±1.2</td><td>2.1 ±1.0</td><td>2.7 ±1.6</td><td>3.5 ±1.7</td><td>N/A</td></tr><tr><td>WPN-CRF</td><td>23.1 ±2.8</td><td>29.9 ±3.2</td><td>32.9 ±1.2</td><td>33.2 ±1.1</td><td>N/A</td></tr><tr><td>Struct NN shot</td><td>31.1 ±6.4</td><td>33.2 ±2.0</td><td>30.8 ±2.2</td><td>31.8 ±1.8</td><td>N/A</td></tr><tr><td>TANL</td><td>25.6 ±6.3</td><td>33.3 ±4.4</td><td>34.1 ±2.1</td><td>34.4 ±2.4</td><td>45.2 ±0.6</td></tr><tr><td>Our model - GloVe</td><td>36.6 ±2.4</td><td>39.6 ±1.9</td><td>42.5 ±1.3</td><td>43.0 ±1.1</td><td>45.7 ±0.6</td></tr><tr><td>Our model - BERT</td><td>38.3 ±1.7</td><td>40.8 ±2.1</td><td>42.7 ±1.1</td><td>43.3 ±0.8</td><td>45.0 ±0.6</td></tr><tr><td rowspan="7">JNLPBA</td><td>TransferBERT</td><td>26.6 ±7.8</td><td>40.3 ±2.8</td><td>53.2 ±2.9</td><td>59.7 ±1.3</td><td>71.0 ±0.5</td></tr><tr><td>Prototypical Network</td><td>2.1 ±1.5</td><td>4.0 ±3.2</td><td>6.8 ±3.6</td><td>5.7 ±3.0</td><td>N/A</td></tr><tr><td>WPN-CRF</td><td>6.5 ±5.0</td><td>10.3 ±5.7</td><td>10.3 ±4.9</td><td>9.4 ±2.7</td><td>N/A</td></tr><tr><td>Struct NN shot</td><td>15.9 ±5.3</td><td>19.2 ±2.9</td><td>23.1 ±2.1</td><td>26.8 ±0.7</td><td>N/A</td></tr><tr><td>TANL</td><td>32.4 ±4.0</td><td>41.1 ±5.0</td><td>51.7 ±2.6</td><td>58.8 ±0.6</td><td>74.3 ±0.2</td></tr><tr><td>Our model - GloVe</td><td>25.4 ±6.1</td><td>39.7 ±2.3</td><td>52.3 ±3.1</td><td>59.3 ±1.4</td><td>71.8 ±0.3</td></tr><tr><td>Our model - BERT</td><td>32.7 ±3.0</td><td>43.15 ±2.4</td><td>53.8 ±2.7</td><td>59.8 ±1.3</td><td>71.0 ±0.5</td></tr><tr><td rowspan="7">NCBI-disease</td><td>TransferBERT</td><td>16.8 ±9.5</td><td>24.1 ±6.3</td><td>43.0 ±5.0</td><td>56.7 ±3.0</td><td>84.5 ±0.9</td></tr><tr><td>Prototypical Network</td><td>12.2 ±8.7</td><td>12.5 ±9.6</td><td>14.0 ±11.6</td><td>10.8 ±7.3</td><td>N/A</td></tr><tr><td>WPN-CRF</td><td>5.5 ±4.8</td><td>6.8 ±9.1</td><td>3.5 ±5.4</td><td>5.7 ±5.3</td><td>N/A</td></tr><tr><td>Struct NN shot</td><td>18.5 ±5.6</td><td>20.6 ±5.2</td><td>27.6 ±2.4</td><td>36.7 ±5.0</td><td>N/A</td></tr><tr><td>TANL</td><td>15.8 ±4.0</td><td>21.0 ±6.2</td><td>26.0 ±3.9</td><td>40.9 ±4.2</td><td>85.8 ±0.9</td></tr><tr><td>Our model - GloVe</td><td>15.1 ±8.7</td><td>26.2 ±6.1</td><td>44.6 ±4.2</td><td>56.8 ±3.1</td><td>86.7 ±0.6</td></tr><tr><td>Our model - BERT</td><td>30.7 ±9.1</td><td>34.9 ±4.9</td><td>50.9 ±3.3</td><td>60.5 ±2.2</td><td>85.0 ±0.6</td></tr><tr><td rowspan="7">12B2-2014</td><td>TransferBERT</td><td>58.4 ±5.7</td><td>75.2 ±1.9</td><td>86.2 ±0.9</td><td>90.3 ±0.4</td><td>93.0 ±0.1</td></tr><tr><td>Prototypical Network</td><td>2.1 ±0.7</td><td>2.2 ±0.4</td><td>2.6 ±0.4</td><td>2.7 ±0.1</td><td>N/A</td></tr><tr><td>WPN-CRF</td><td>10.0 ±2.5</td><td>13.1 ±3.3</td><td>13.9 ±2.1</td><td>13.3 ±2.1</td><td>N/A</td></tr><tr><td>Struct NN shot</td><td>46.7 ±6.4</td><td>59.1 ±1.9</td><td>67.4 ±1.3</td><td>72.4 ±0.6</td><td>N/A</td></tr><tr><td>TANL</td><td>47.1 ±5.2</td><td>65.1 ±2.9</td><td>80.7 ±1.2</td><td>87.0 ±0.3</td><td>92.0 ±0.1</td></tr><tr><td>Our model - GloVe</td><td>58.2 ±5.8</td><td>75.5 ±2.3</td><td>85.6 ±1.0</td><td>90.5 ±0.3</td><td>93.5 ±0.1</td></tr><tr><td>Our model - BERT</td><td>61.9 ±4.3</td><td>76.8 ±2.0</td><td>86.7 ±0.8</td><td>90.5 ±0.4</td><td>93.2 ±0.3</td></tr></table>


Table 1: Results on held out test sets of all datasets. "Our model - GloVe": this refers to our model with GloVe label encoder. "Our model - BERT": this refers to our model with BERT label encoder. All numbers indicate micro F1 scores unless noted otherwise. Results for low resource settings are average of 10 runs with different support set sampling. Results for high resource setting are average of 5 runs with different random seeds. For some baselines we cannot run the released implementation from originally papers due to GPU out of memory and they are marked as N/A. We visualize the results with bar chart in Appendix D.


Low Resource: Given a target dataset, we downsample the data (at sentence level) in the train split to construct a $K$ -shot support set. This simulates the low resource scenario where only a few training examples are available in the target dataset. The definition of a $K$ -shot support set is that it contains exact $K$ examples for each of the labels. However, unlike the text classification task where each sen 

tence is associated with one label, in the NER task multiple named entities may co-occur in the same sentence. We cannot guarantee that the support set contains exact $K$ named entities for each label after downsampling. We therefore define the proxy for $K$ -shot support set similar as the one by Hou et al. (2020), with the following two criteria: 1) Each label in the target dataset (except "O") has at least 

$K$ corresponding named entities in the support set; 2) At least one of the labels in the target dataset will have less than $K$ named entities in the support set if any sentence is removed. We apply the same downsampling algorithm as in (Hou et al., 2020) for the support set. More details can be found in Appendix B. 

To evaluate the model performance in the $K$ -shot support set, most prior work (Hou et al., 2020; Athiwaratkun et al., 2020; Fritzler et al., 2019) followed the few-shot classification setup, where test sets are also downsampled to $K$ -shot subsets (query set) such that each entity labels are evenly distributed. The model is trained and evaluated on multiple support datasets and query set pairs, and final model performance is reported with average of scores on each query set. However, we argue that in real world cases, entity labels have certain distribution corresponding to the domain, downsampled $K$ -shot query set does not reflect this real distribution. Therefore instead of evaluating on the downsampled query set, we directly evaluate the model in the full test split from the target dataset. This also improves comparability and replicability of our results since the same test set is used across and in prior work (even in papers that are not focused on few-shot experiments). 

Evaluation To thoroughly test our model, we evaluate it with 1-shot, 5-shot, 20-shot, 50-shot (low resource) and also the full dataset (high resource) settings. Following prior work (Tjong Kim Sang and De Meulder, 2003b), we use micro F1 score as metric. For low resource settings, we repeat the experiments 10 times with randomly sampled support sets. For high resource setting, we repeat the experiments 5 times with different random seeds. In all cases, we report average micro F1 with standard deviation. Table 2 shows an overview of dataset statistics. 

# 3.3 Baselines

TransferBERT trains the same NER model in (Devlin et al., 2019) by pre-finetuning on a source dataset then finetuning on a target dataset. Proto 

typical Network (Snell et al., 2017) approaches NER as a token level classification task. It assigns label for each token based on similarities between candidate token and tokens in few shot support set. WPN-CRF (Fritzler et al., 2019) pretrains a prototypical network with source dataset and evaluate it on target dataset without finetuning. It uses a conditional random field (CRF) (Huang et al., 2015) to output the final labels of the sentence. Struct NN shot (Yang and Katiyar, 2020) finds nearest token in support set for a given candidate token and assign it the same label as its nearest neighbor. TANL (Paolini et al., 2021) forms NER as sequence to sequence. The model is trained to generate the original input text with entities being decorated in a bracket. $^6$ 

# 3.4 Hyperparameters

We use English cased BERT-base (Devlin et al., 2019) as contextual embedder for all baseline models and our model, except for TANL where T5-base is used. We use Adam optimizer (Kingma and Ba, 2014) to train our model with a learning rate of $1 \times 10^{-5}$ and batch size of 10. We pre-finetune our model on the source dataset (Ontonotes) for 3 epochs and continue finetuning on target datasets for 200 epochs for both high resource and low resource settings. We pick the last epoch as the final model. For label names, we manually expand all shortcut names into full natural language names (e.g. "PER" to "person", "LOC" to "location") and lower case all names. Textual forms for all datasets can be found in Appendix A.2. We run all experiments on NVIDIA V100 GPU. 

# 3.5 GloVe as Label Encoder

We experiment with GloVe embeddings (Pennington et al., 2014) as the label encoder. In this case, 

our model has no extra parameters compared to other baselines. As in the case with BERT, the vectors are updated throughout the training. Given that there is no [CLS] token available, we apply max pooling on all the GloVe embeddings corresponding to each label token. If the label consists only of one token, max pooling will return the actual GloVe embedding for the token as the label representation. 

<table><tr><td rowspan="2">Dataset</td><td colspan="4">Support Set Shot</td></tr><tr><td>1</td><td>5</td><td>20</td><td>50</td></tr><tr><td>CoNLL&#x27;03</td><td>3.6</td><td>12.3</td><td>38.5</td><td>102.5</td></tr><tr><td>WNUT&#x27;17</td><td>13.4</td><td>44.6</td><td>143.6</td><td>366.3</td></tr><tr><td>JNLPBA</td><td>6.8</td><td>27.5</td><td>99.2</td><td>241.2</td></tr><tr><td>NCBI</td><td>1.8</td><td>3.7</td><td>14.5</td><td>37.2</td></tr><tr><td>I2B2&#x27;14</td><td>155.4</td><td>613.4</td><td>2339.4</td><td>5888.1</td></tr></table>


Table 2: Number of sentences in support set with different shots for all target datasets. Numbers are averaged across 10 different random samplings. NCBI refers to NCBI-disease dataset. More details are reported in Appendix A.1.


# 3.6 Results

We summarize experiment results in Table 1. As shown, our model outperforms all previous methods in low resource settings. In extreme low resource scenarios (1 and 5 shot), our model performs significantly better than previous methods by a margin of 6.6 F1 and 4.8 F1 on average in 1 shot and 5 shot, respectively. This indicates that our model can leverage semantics in label names effectively to improve accuracy when data is extremely scarce. However, we also notice that when the target data size increases, the improvement of our model becomes smaller. This suggests that with more training examples, the model relies less on semantics of labels. 

In a high resource setting, we find that our model achieves the same level of performance as other baselines, except for JNLPBA dataset where our model is 3.3 F1 behind TANL. $^{10}$ This model is based on T5-base which is pretrained on a much 

larger unannotated dataset, and with different objectives, than our BERT-base encoders. 

We also note that when label names in the target dataset are similar to the source ones, few shot models have a much smaller gap with their high resource counterparts, compared to when source and target label names are totally different. Specifically, CoNLL-2003, WNUT-2017 and I2B2 have more similar label names with Ontonotes (the source data), and our model can achieve $84\%$ , $91\%$ and $83\%$ of the score of the high resource model performance with only 5 shot. While for JNLPBA and NCBI-disease, where the label names are totally different from source data, our model can only achieve $61\%$ and $41\%$ of the score of the high resource model performance with 5 shot. 

# 4 Analysis

Here, we show how semantics in label names help in low resource scenarios and how our model benefits from pre-finetuning stage. 

<table><tr><td rowspan="2">Entity Types</td><td colspan="2">Original Labels</td><td>Renamed Labels</td></tr><tr><td>0 shot</td><td>1 shot</td><td>0 shot</td></tr><tr><td>PER</td><td>92.3</td><td>90.3</td><td>85.4</td></tr><tr><td>LOC</td><td>70.9</td><td>61.2</td><td>54.8</td></tr><tr><td>ORG</td><td>50.3</td><td>59.7</td><td>58.4</td></tr><tr><td>MISC</td><td>0.5</td><td>47.5</td><td>6.8</td></tr></table>


Table 3: F1 for 0 and 1 shot performance on CoNLL-2003 development set.


# 4.1 Impact of the Label Encoder

We hypothesize that encoding label names with a label encoder (either BERT or GloVe) leverages prior knowledge from the pretraining phase and uses it as inductive bias. In addition, by performing pre-finetuning on the source dataset, we are not only aligning the embedding space between labels and tokens in the vocabulary, but also updating the label encoder to produce useful label representations in the source dataset. 

To further strengthen our hypothesis (besides what is presented in Table 1), we show results in zero shot settings. Specifically, we pre-finetune a model on the source dataset (Ontonotes) and directly test it on CoNLL-2003 without updating its parameters. We also rename the labels to avoid 

overlapping of label names between source and target datasets while still retaining the semantics. $^{11}$ Particularly, during evaluation we rename “PER” to “individual”, “LOC” to “geographical area” and “ORG” to “corporation”. “MISC” stays the same since it does not overlap with any of the Antonotes labels. The results are shown in Table 3. 

With original label names, the zero shot performance of our model is comparable to 1 shot performance for all entity types with the exception of "MISC". Even with the renamed labels that do not have any overlap with the source dataset, the zero shot performance still remains comparable with 1 shot. This seems to validate our hypothesis that the model is able to leverage prior knowledge. 

# 4.2 Semantics of Label Names

To demonstrate the impact of semantics of label names, we carry out experiments with our model on target datasets with the following variations of label names: (1) original label names (which is simply our experimental setup as in the experiments above, where we use the natural language form of the label names), (2) meaningless label names and (3) misleading label names. 

We compare our model with the TransferBERT baseline, since it is the counterpart of our model without label semantics. We pre-finetune our model on Ontonotes as previous experiments. Results on CoNLL2003 and JNLPBA are shown in Figure 2. $^{12}$ 

Meaningless labels We simply use "label 1", "label 2" etc., as input representation for label names, which simulates the case where there is no more semantics information in the form than the fact that they are different labels and they have some sort of ordering. This evaluates the few shot model performance when meaningless (or shallow in semantics, just a differentiation of label indices) inputs are given. Comparing to the original label names, the results drop in 1 and 5 shot settings, then gradually converged to the original label performance as the training data size increases. This shows that 

label semantics is critical for extreme low resource scenarios (1 and 5 shot). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-27/193d6cc2-1574-4554-b7cb-92d5b465075b/d9c94dba6682f016ddfa5fd39328095cd2e192e815824f18ce5247ed0a84d832.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-27/193d6cc2-1574-4554-b7cb-92d5b465075b/e32777c18d3624ada820c188ee4579fc1e5a65549ce19a10d01f27fcac3ed075.jpg)



Figure 2: Model performance on meaningless and misleading labells. Micro F1 is reported on the development data.


Misleading labels We randomly swap the natural language form between labels. For example, in CoNLL2003 dataset, we assign "location" for "PER", "person" for "ORG", "organization" for "MISC" and "miscellaneous" for "PER". The performance drops are larger for CoNLL2003 than the ones in JNLPBA. We hypothesize that since CoNLL2003 label set is closer to Antonotes, there is stronger prior knowledge incorporated in the label encoder from the pre-finetuning phase. Also, we find that more supervised examples are required to correct such wrong strong prior information. JNLPBA needs 5 shot data to achieve the same performance with original labels and misleading labels, but CoNLL2003 needs 50 shot data to match the performance. This indicates that our model is misled by the labels when the number of training examples is small, which indicates that the label semantics signal is critical in few shot settings. 

# 4.3 Impact of Pre-finetuning

Our model does not require a new randomly initialized top layer classifier for a new dataset, we hypothesize that it can prevent the model from forgetting learned prior knowledge from the prefinetuning stage thus benefits the low resource scenarios, where prior knowledge is critical. To validate it, we compare 1-shot results on target datasets with and without pre-finetuning stage, as shown in Table 4. First, when pre-finetuning stage is eliminated, performance of both our model and TransferBERT drop significantly, indicating that prior knowledge from pre-finetuning stage is critical in low resource settings. Second, our model outperforms TransferBERT significantly when pre-finetuning stage is included, however, the performance is similar between our model and TransferBERT when it is excluded. This suggests that our model is highly effective in leveraging knowledge learned from the pre-finetuning stage. 

<table><tr><td rowspan="2">Datasets</td><td colspan="2">Pre-finetune on Ontonotes</td><td colspan="2">No pre-finetune</td></tr><tr><td>Transfer-BERT</td><td>Ours</td><td>Transfer-BERT</td><td>Ours</td></tr><tr><td>CoNLL&#x27;03</td><td>47.5</td><td>69.0</td><td>9.0</td><td>10.7</td></tr><tr><td>WNUT&#x27;17</td><td>35.6</td><td>48.2</td><td>4.0</td><td>5.7</td></tr><tr><td>JNLPBA</td><td>26.3</td><td>31.5</td><td>14.8</td><td>19.5</td></tr><tr><td>NCBI</td><td>15.1</td><td>31.3</td><td>12.5</td><td>13.9</td></tr><tr><td>I2B2&#x27;14</td><td>56.9</td><td>60.1</td><td>47.5</td><td>46.8</td></tr></table>


Table 4: 1-shot performance on development set of corresponding datasets. Micro F1 is reported. NCBI refers to NCBI-disease dataset.


# 5 Related Work

Few Shot Learning: Meta learning is widely studied for the problem of few shot learning, aiming to quickly adapt a model to new tasks based on tasks learned in an earlier stage. Recent research (Snell et al., 2017; Vinyals et al., 2017; Sung et al., 2017) mostly focused on metric-based methods. Snell et al. (2017) learns a prototype representation for each class and classify test data based on their similarities with prototypes. These methods have been successfully adapted to NLP tasks such as classification (Yu et al., 2018b; Bao et al., 2019), relation classification (Han et al., 2018) and NER (Fritzler et al., 2019; Yang and Katiyar, 2020). 

However, all these methods do not directly leverage the semantics of label names. 

Label Semantics: Earlier work has shown the ability to perform zero- and few-shot learning by exploiting the semantic of labels in text classification tasks (Chang et al., 2008; Luo et al., 2021). Zhou et al. (2018) study zero-shot fine-type NER with label semantics by automatically reading from Wikipedia via a linking approach, but assumes that the mentions of the entities are given. Paolini et al. (2021) and Athiwaratkun et al. (2020) approach NER as a generation task and predict named entities in augmented (or decorated) languages. Cui et al. (2021) reformulate NER as a cloze task and use sequence to sequence models to fill named entities in pre-defined templates. Both of these two methods suffer from long inference time due to an autoregressive decoder. Hou et al. (2020) leverage label semantics in Task-Adaptive Projection Network (TapNet), where the core idea is to learn a projection function that separates words that have different labels in the projected space. In contrast, our model learns to align token representations with label representations. Hou et al. (2020) only uses label representations as a reference to guide the learning of the projection function, and in their case label representations are computed once. Our label representations are updated with every update while training. 

# 6 Conclusion

We propose a neural architecture that leverages semantics of label names for Named Entity Recognition. Our model significantly outperforms the state-of-the-art few shot NER baselines on low resource settings, and performs on-par in the high resource setting. We perform extensive experiments to show that the label encoder incorporates strong prior knowledge from BERT and a dataset (source dataset) used in a pre-finetuning stage. We demonstrate that the semantics of label names in target datasets are critical to retrieve the prior knowledge. We also show that our model is robust to variation of label names and that it is able to differentiate between semantically closed labels. 

# References



Armen Aghajanyan, Anchit Gupta, Akshit Shrivastava, Xilun Chen, Luke Zettlemoyer, and Sonal Gupta. 2021. Muppet: Massive multi-task representations with pre-finetuning. CoRR, abs/2101.11038. 





Ben Athiwaratkun, Cicero Nogueira dos Santos, Jason Krone, and Bing Xiang. 2020. Augmented natural language for generative sequence labeling. 





Yujia Bao, Menghua Wu, Shiyu Chang, and Regina Barzilay. 2019. Few-shot text classification with distributional signatures. CoRR, abs/1908.06039. 





Xavier Carreras, Lluís Márquez, and Lluís Padró. 2003. Learning a perceptron-based named entity chunker via online recognition feedback. In Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003, pages 156-159. 





Yee Seng Chan and Dan Roth. 2011. Exploiting syntactico-semantic structures for relation extraction. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 551-560, Portland, Oregon, USA. Association for Computational Linguistics. 





Ming-Wei Chang, Lev-Arie Ratinov, Dan Roth, and Vivek Srikumar. 2008. Importance of semantic representation: Dataless classification. In AAAI. 





Jiaao Chen, Zhenghui Wang, Ran Tian, Zichao Yang, and Diyi Yang. 2020. Local additivity based data augmentation for semi-supervised NER. CoRR, abs/2010.01677. 





Kevin Clark and Christopher D. Manning. 2015. Entity-centric coreference resolution with model stacking. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1405-1415, Beijing, China. Association for Computational Linguistics. 





Nigel Collier and Jin-Dong Kim. 2004. Introduction to the bio-entity recognition task at JNLPBA. In Proceedings of the International Joint Workshop on Natural Language Processing in Biomedicine and its Applications (NLPBA/BioNLP), pages 73-78, Geneva, Switzerland. COLING. 





Ronan Collobert, Jason Weston, León Bottou, Michael Karlen, Koray Kavukcuoglu, and Pavel P. Kuksa. 2011. Natural language processing (almost) from scratch. CoRR, abs/1103.0398. 





Leyang Cui, Yu Wu, Jian Liu, Sen Yang, and Yue Zhang. 2021. Template-based named entity recognition using BART. CoRR, abs/2106.01760. 





Leon Derczynski, Eric Nichols, Marieke van Erp, and Nut Limsopatham. 2017. Results of the WNUT2017 shared task on novel and emerging entity recognition. In Proceedings of the 3rd Workshop on Noisy User-generated Text, pages 140-147, Copenhagen, Denmark. Association for Computational Linguistics. 





Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. 





Ning Ding, Guangwei Xu, Yulin Chen, Xiaobin Wang, Xu Han, Pengjun Xie, Hai-Tao Zheng, and Zhiyuan Liu. 2021. Few-nerd: A few-shot named entity recognition dataset. CoRR, abs/2105.07464. 





Rezarta Islamaj Dogan, Robert Leaman, and Zhiyong Lu. 2014. Ncbi disease corpus: A resource for disease name recognition and concept normalization. Journal of biomedical informatics, 47:1-10. 





Chelsea Finn, Pieter Abbeel, and Sergey Levine. 2017. Model-agnostic meta-learning for fast adaptation of deep networks. 





Alexander Fritzler, Varvara Logacheva, and Maksim Kretov. 2019. Few-shot classification in named entity recognition task. Proceedings of the 34th ACM/SIGAPP Symposium on Applied Computing. 





Xu Han, Hao Zhu, Pengfei Yu, Ziyun Wang, Yuan Yao, Zhiyuan Liu, and Maosong Sun. 2018. Fewrel: A large-scale supervised few-shot relation classification dataset with state-of-the-art evaluation. CoRR, abs/1810.10147. 





Yutai Hou, Wanxiang Che, Yongkui Lai, Zhihan Zhou, Yijia Liu, Han Liu, and Ting Liu. 2020. Few-shot slot tagging with collapsed dependency transfer and label-enhanced task-adaptive projection network. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1381-1393, Online. Association for Computational Linguistics. 





Jiaxin Huang, Chunyuan Li, Krishan Subudhi, Damien Jose, Shobana Balakrishnan, Weizhu Chen, Baolin Peng, Jianfeng Gao, and Jiawei Han. 2020. Few-shot named entity recognition: A comprehensive study. CoRR, abs/2012.14978. 





Zhiheng Huang, Wei Xu, and Kai Yu. 2015. Bidirectional lstm-crf models for sequence tagging. 





Vladimir Karpukhin, Barlas Oguz, Sewon Min, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. CoRR, abs/2004.04906. 





Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980. 





Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. 2016. Neural architectures for named entity recognition. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 260-270, San Diego, California. Association for Computational Linguistics. 





Pierre Lison, Jeremy Barnes, Aliaksandr Hubin, and Samia Touileb. 2020. Named entity recognition without labelled data: A weak supervision approach. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1518-1533, Online. Association for Computational Linguistics. 





Lajanugen Logeswaran, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, Jacob Devlin, and Honglak Lee. 2019. Zero-shot entity linking by reading entity descriptions. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3449-3460, Florence, Italy. Association for Computational Linguistics. 





Qiaoyang Luo, Lingqiao Liu, Yuhao Lin, and Wei Zhang. 2021. Don't miss the labels: Label-semantic augmented meta-learner for few-shot text classification. In FINDINGS. 





Diego Mollá, Menno van Zaanen, and Daniel Smith. 2006. Named entity recognition for question answering. In Proceedings of the Australasian Language Technology Workshop 2006, pages 51-58, Sydney, Australia. 





Giovanni Paolini, Ben Athiwaratkun, Jason Krone, Jie Ma, Alessandro Achille, Rishita Anubhai, Cicero Nogueira dos Santos, Bing Xiang, and Stefano Soatto. 2021. Structured prediction as translation between augmented natural languages. 





Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pages 1532-1543. 





Jake Snell, Kevin Swersky, and Richard S. Zemel. 2017. Prototypical networks for few-shot learning. 





Amber Stubbs and Ozlem Uzuner. 2015. Annotating longitudinal clinical narratives for de-identification. J. of Biomedical Informatics, 58(S):S20-S29. 





Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip H. S. Torr, and Timothy M. Hospedales. 2017. Learning to compare: Relation network for few-shot learning. CoRR, abs/1711.06025. 





Erik F. Tjong Kim Sang and Fien De Meulder. 2003a. Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. In Proceedings of the Seventh Conference on Natural 





Language Learning at HLT-NAACL 2003, pages 142-147. 





Erik F. Tjong Kim Sang and Fien De Meulder. 2003b. Introduction to the conll-2003 shared task: Language-independent named entity recognition. In Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003 - Volume 4, CONLL '03, page 142-147, USA. Association for Computational Linguistics. 





Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, and Daan Wierstra. 2017. Matching networks for one shot learning. 





Yogarshi Vyas and Miguel Ballesteros. 2020. Linking entities to unseen knowledge bases with arbitrary schemas. CoRR, abs/2010.11333. 





Tian Wang, Yuri M. Brovman, and Sriganesh Madhavanath. 2021. Personalized embedding-based e-commerce recommendations at ebay. CoRR, abs/2102.06156. 





Ralph Weischedel, Martha Palmer, Mitchell Marcus, Eduard Hovy, Sameer Pradhan, Lance Ramshaw, Nianwen Xue, Ann Taylor, Jeff Kaufman, Michelle Franchini, et al. 2013. Ontonotes release 5.0 ldc2013t19. Linguistic Data Consortium, Philadelphia, PA, 23. 





Yi Yang and Arzoo Katiyar. 2020. Simple and effective few-shot named entity recognition with structured nearest neighbor learning. 





Mo Yu, Xiaoxiao Guo, Jinfeng Yi, Shiyu Chang, Saloni Potdar, Yu Cheng, Gerald Tesauro, Haoyu Wang, and Bowen Zhou. 2018a. Diverse few-shot text classification with multiple metrics. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1206-1215, New Orleans, Louisiana. Association for Computational Linguistics. 





Mo Yu, Xiaoxiao Guo, Jinfeng Yi, Shiyu Chang, Saloni Potdar, Yu Cheng, Gerald Tesauro, Haoyu Wang, and Bowen Zhou. 2018b. Diverse few-shot text classification with multiple metrics. CoRR, abs/1805.07513. 





Ben Zhou, Daniel Khashabi, Chen-Tse Tsai, and Dan Roth. 2018. Zero-shot open entity typing as type-compatible grounding. In EMNLP. 



# A Datasets Details

# A.1 Statistics

Table 5 shows the statistics of original datasets we use in the main experiments. 

<table><tr><td>Dataset</td><td>Domain</td><td># Sent</td><td># Labels</td></tr><tr><td>Ontonotes</td><td>Mix</td><td>76,714</td><td>18</td></tr><tr><td>CoNLL’03</td><td>News</td><td>20,744</td><td>4</td></tr><tr><td>WNUT’07</td><td>Social</td><td>5,690</td><td>6</td></tr><tr><td>JNLPBA</td><td>Bio</td><td>22,402</td><td>5</td></tr><tr><td>NCBI-disease</td><td>Bio</td><td>7,287</td><td>1</td></tr><tr><td>I2B2’14</td><td>Medical</td><td>75,330</td><td>23</td></tr></table>


Table 5: Original dataset statistics.


<table><tr><td>Dataset</td><td>Original
Labels</td><td>Natural
Language</td></tr><tr><td rowspan="4">CoNLL&#x27;03</td><td>PER</td><td>person</td></tr><tr><td>LOC</td><td>location</td></tr><tr><td>ORG</td><td>organization</td></tr><tr><td>MISC</td><td>miscellaneous</td></tr><tr><td rowspan="19">Ontonotes</td><td>CARDINAL</td><td>cardinal</td></tr><tr><td>DATE</td><td>date</td></tr><tr><td>EVENT</td><td>event</td></tr><tr><td>FAC</td><td>facility</td></tr><tr><td>GPE</td><td>geographical social</td></tr><tr><td>LANGUAGE</td><td>political entity</td></tr><tr><td>LAW</td><td>language</td></tr><tr><td>LOC</td><td>law</td></tr><tr><td>MONEY</td><td>location</td></tr><tr><td>NORP</td><td>money</td></tr><tr><td>ORDINAL</td><td>nationality religion</td></tr><tr><td>ORG</td><td>political</td></tr><tr><td>PERCENT</td><td>ordinal</td></tr><tr><td>PERSON</td><td>organization</td></tr><tr><td>PRODUCT</td><td>percent</td></tr><tr><td>QUANTITY</td><td>product</td></tr><tr><td>TIME</td><td>quantity</td></tr><tr><td>WORK_OF_ART</td><td>time</td></tr><tr><td>corporation</td><td>work of art</td></tr><tr><td rowspan="6">WNUT&#x27;17</td><td>corporation</td><td>corporation</td></tr><tr><td>creative-work</td><td>creative work</td></tr><tr><td>group</td><td>group</td></tr><tr><td>location</td><td>location</td></tr><tr><td>person</td><td>person</td></tr><tr><td>product</td><td>product</td></tr><tr><td rowspan="5">JNLPBA</td><td>DNA</td><td>DNA</td></tr><tr><td>RNA</td><td>RNA</td></tr><tr><td>cell_line</td><td>cell line</td></tr><tr><td>cell_type</td><td>cell type</td></tr><tr><td>protein</td><td>protein</td></tr><tr><td>NCBI-disease</td><td>Disease</td><td>disease</td></tr><tr><td rowspan="23">I2B2&#x27;14</td><td>AGE</td><td>age</td></tr><tr><td>BIOID</td><td>biometric ID</td></tr><tr><td>CITY</td><td>city</td></tr><tr><td>COUNTRY</td><td>country</td></tr><tr><td>DATE</td><td>date</td></tr><tr><td>DEVICE</td><td>device</td></tr><tr><td>DOCTOR</td><td>doctor</td></tr><tr><td>EMAIL</td><td>email</td></tr><tr><td>FAX</td><td>fax</td></tr><tr><td>HEALTHPLAN</td><td>health plan number</td></tr><tr><td>HOSPITAL</td><td>hospital</td></tr><tr><td>IDNUM</td><td>ID number</td></tr><tr><td>LOCATION_OTHER</td><td>location</td></tr><tr><td>MEDICALRECORD</td><td>medical record</td></tr><tr><td>ORGANIZATION</td><td>organization</td></tr><tr><td>PATIENT</td><td>patient</td></tr><tr><td>PHONE</td><td>phone number</td></tr><tr><td>PROFESSION</td><td>profession</td></tr><tr><td>STATE</td><td>state</td></tr><tr><td>STREET</td><td>street</td></tr><tr><td>URL</td><td>url</td></tr><tr><td>USERNAME</td><td>username</td></tr><tr><td>ZIP</td><td>zip code</td></tr></table>

# A.2 Label Names


Table 6 shows the original label names in each dataset and corresponding natural language forms we use in our experiments.



Table 6: Original label names and their corresponding natural language formats.


# B Support Set Sampling Algorithm


Algorithm 1 Support set sampling


Require: # shot $K$ , dataset $\mathcal{D}$ , labels $\mathcal{L}_{\mathcal{D}}$ 1: Initialize support set $\mathcal{S} = \{\}$ , $\mathrm{Count}_{\ell_i} = 0$ ( $\forall \ell_i \in \mathcal{L}_{\mathcal{D}}$ )  
2: for $\ell$ in $\mathcal{L}_{\mathcal{D}}$ do  
3: while $\mathrm{Count}_{\ell} < K$ do  
4: Randomly pick $(t, y)$ from $\mathcal{D} \setminus \mathcal{S}$ that $y$ include $\ell$ 5: $\mathcal{S} \gets \mathcal{S} \cup (t, y)$ 6: Update all $\mathrm{Count}_{\ell_i} (\forall \ell_i \in \mathcal{L}_{\mathcal{D}})$ 7: end while  
8: end for  
9: for $(t, y)$ in $\mathcal{S}$ do  
10: $\mathcal{S} = \mathcal{S} \setminus (t, y)$ 11: Update all $\mathrm{Count}_{\ell_i} (\forall \ell_i \in \mathcal{L}_{\mathcal{D}})$ 12: if Any $\mathrm{Count}_{\ell_i} < K$ then  
13: $\mathcal{S} = \mathcal{S} \cup (t, y)$ 14: Update all $\mathrm{Count}_{\ell_i} (\forall \ell_i \in \mathcal{L}_{\mathcal{D}})$ 15: end if  
16: end for 

# C Hardware for Experiments

We provide details about hardware we use to produce numbers for each baseline models. We run experiments for Struct NN shot model on NVIDIA V100 GPU with 32GB of memory, while for all other models (including baselines and our models) we use NVIDIA V100 GPU with 16GB of memory. 

# D Visualization of Results

We visualize the results in Table 1 with bar chart, as shown in Figure 3. 

# E Contextualized Label Representations

In this experiment, we compute contextualized label representations by randomly selecting a sentence from the support set that contains an entity of the type, and replace that entity with the label name in the sentence. We encode this sentence with the label encoder and compute the average pooling as the label representation. The label names used are in their natural language form with BIO schemes per 2.2. We depict this process in Figure 4. At inference time, to avoid biasing toward any particular sentence, we randomly choose 10 sentences from the support set for each label and average their representations as the final label representations.[14] 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-27/193d6cc2-1574-4554-b7cb-92d5b465075b/4a1219d25be9024597c6a07fb983fa5360e037eadb26f8535c2eb78b2ac43554.jpg)



Figure 4: Differences between contextualized label representations and label representations in isolation.


We perform experiments on FEW-NERD dataset (Ding et al., 2021). This dataset consists of 8 coarse-grained and 66 fine-grained entity types in hierarchy. The fine-grained entity types under the same coarse-grained type are semantically close. 

Results are shown in Table 7 and Appendix E. In the following, we show 1-shot results under "Person" coarse-grained type for FEW-NERD dataset.[16] By using contextual label names, we observe a decrease in model performance by 3.5 F1 points on FEW-NERD, compared to when only label names are used. This suggests that the trained label encoder is capable of capturing critical semantics with only label names, even without contexts to help distinguish semantically close labels. 

<table><tr><td rowspan="2">Datasets</td><td colspan="2">Model</td></tr><tr><td>Ours</td><td>Ours + context</td></tr><tr><td>CoNLL&#x27;03</td><td>69.0±6.9</td><td>70.8±4.1</td></tr><tr><td>WNUT17</td><td>48.2±1.7</td><td>51.8±1.8</td></tr><tr><td>JNLPBA</td><td>31.5±2.9</td><td>30.1±3.2</td></tr><tr><td>FEW-NERD-Person</td><td>32.5±8.1</td><td>29.0±7.1</td></tr></table>


Table 7: 1-shot micro F1 on development set across various datasets and models. Ours: Our model with label names. Ours+context: Our model with contextual label names. Numbers are averaged across 10 different random samplings.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-27/193d6cc2-1574-4554-b7cb-92d5b465075b/7ac59ae23cec820e3087fa9822be9b9fe809f26b2650ce518800923eb868a50c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-27/193d6cc2-1574-4554-b7cb-92d5b465075b/9ff504b35dc4b392dd63982be7049e3c42226e7405d633e688a047004391138e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-27/193d6cc2-1574-4554-b7cb-92d5b465075b/59faeea1f4fc0092fbf5e582a51a0d14f1e006ac50a9b607936ae503d4bbdc49.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-27/193d6cc2-1574-4554-b7cb-92d5b465075b/782eb942eb36c37dd1b22e576bf4d933b26a3f9c919d7fab024c7ee9ca677f93.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-27/193d6cc2-1574-4554-b7cb-92d5b465075b/960918d07d181203e6479591028ace59ff2e8996ca15a8369a3379eeffa7c1af.jpg)



Figure 3: Visualization of the results in Table 1. Results on test set of all datasets. All numbers indicate micro F1 scores except noted otherwise. Results for low resource settings are average of 10 runs with different support set sampling. Results for high resource setting are average of 5 runs with different random seeds. For some baselines we cannot run the released implementation from originally papers due to GPU out of memory and they are marked as 0.


# E.1 Additional Experiment 1

We present additional experiments on contextual label representations. We will first introduce more details on the FEW-NERD dataset, then describe methods we explore to contextualize labels, finally we will show experiment results. To validate whether contextual label representation can improve model performance in scenarios where labels are semantically close, we perform experiments on one additional dataset: FEW-NERD (Ding et al., 2021). FEW-NERD is a human annotated NER dataset that consists of 188,238 sentences. It has a hierarchy of 8 coarse-grained and 66 fine-grained entity types. The fine-grained entity types under each coarse-grained type are usually semantically close. All sentences are sourced from Wikipedia. We use train/dev/test split from the original dataset distribution. 

We select "Person" and "Art" coarse-grained entity types for the experiments, because we think fine-grained entity types under them have closest semantic similarities. Specifically, we take one coarse-grained entity type at a time, and remove all entity annotations that do not belong to it, on train, dev and test split. After removal, comparing with the original dataset, the resulting dataset has much more sentences with no annotation than sentences that have at least one annotations. To mitigate this entity distribution shifting, we randomly remove sentences that do not contain any annotations, such that the resulting dataset has the same percentage of sentences with annotations as the original dataset. We perform this process on "Person" and "Art" types and result in two datasets called "FEW-NERD-Person" and "FEW-NERD-Art". The statistics for these two datasets are shown in Table 8. The original entity types and their corresponding natural language format are shown in Table 9 

<table><tr><td>Dataset</td><td>Original
Labels</td><td>Natural
Language</td></tr><tr><td rowspan="7">FEW-NERD-
Person</td><td>person-actor</td><td>actor</td></tr><tr><td>person-artist/author</td><td>artist author</td></tr><tr><td>person-athlete</td><td>athlete</td></tr><tr><td>person-director</td><td>director</td></tr><tr><td>person-politician</td><td>politician</td></tr><tr><td>person-scholar</td><td>scholar</td></tr><tr><td>person-soldier</td><td>soldier</td></tr><tr><td rowspan="5">FEW-NERD-
Art</td><td>art-broadcastprogram</td><td>broadcast-program</td></tr><tr><td>art-film</td><td>film</td></tr><tr><td>art-music</td><td>music</td></tr><tr><td>art-painting</td><td>painting</td></tr><tr><td>art-written</td><td>written art</td></tr></table>


Table 9: Original label names and their corresponding natural language formats for FEW-NERD-Person and FEW-NERD-Art datasets.


# E.2 Additional Experiment 2

In this experiment, we replace the entity in the selected sentence with different texts rather than label names. 

We experiment with various schemes for the new span and use the following terminology to describe them. $TOKEN$ refers to the original token that is replaced. $LABEL$ refers to the label name that the token is annotated with. $BIO-TAG$ refers to the natural BIO tag that the token is annotated with. For the example illustrated in Figure 4, $TOKEN$ corresponds to "Messi", $LABEL$ corresponds to "person", $BIO-TAG$ corresponds to "begin". We hypothesize that the $TOKEN$ gives natural context to the labels since it is unmodified sentence, $LABEL$ captures the semantic information in label names and $BIO-TAG$ helps differentiate the B and I chunks for the label. In addition, we experiment to replace the entity with "[MASK]" token to make the label reprensetation close to BERT pretraining inputs. The various schemes are illustrated with example in Figure 5. 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2"># Labels</td><td colspan="4">Support Set Shot</td><td rowspan="2">Dev</td></tr><tr><td>1</td><td>5</td><td>20</td><td>50</td></tr><tr><td>FEW-NERD-Person</td><td>7</td><td>19.0</td><td>66.7</td><td>212.7</td><td>508.9</td><td>4437.0</td></tr><tr><td>FEW-NERD-Art</td><td>5</td><td>41.5</td><td>123.5</td><td>412.2</td><td>2569.0</td><td>1364.0</td></tr></table>


Table 8: Number of sentences in support set and dev set for FEW-NERD-Person and FEW-NERD-Art datasets. Numbers are averaged across 10 different random samplings.


# Contextual Label Names Variation Examples

1. Randomly selected sentence from support set: 

"Messi is a soccer player" 

2. Calculate contextual label representation: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-27/193d6cc2-1574-4554-b7cb-92d5b465075b/0e36d34409acacd4915bf09959384cf75b6778f479d2f402c3e90dd0e844d495.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-27/193d6cc2-1574-4554-b7cb-92d5b465075b/a0fdbbb429014930d8b56e9b158e6bf84a401fe3769f2f78d92ef2a74596cdaa.jpg)


: Average pooling 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-27/193d6cc2-1574-4554-b7cb-92d5b465075b/2a1fd39801ec192892c0b022dd5ffa6fa6730ee656d6c73ccd2a9aea1460e492.jpg)


: All tokens encoded by label encoder 

replaced token is same for both B and I chunks in BIO scheme. For example, to get contextualized representation for B-PER in the document "Lionel Messi is a soccer player", the document will be transformed to "person person is a soccer player", where B and I chunks are confused. "BIO-TAG: LABEL" scheme addresses this by prefixing the natural language BIO chunk name to the label name. We see improvements in performance compared with LABEL scheme. 

When we incorporate the “[MASK]” token from BERT pretraining, we find that this does not perform as well as other schemes that contains label names. This further proves that semantics in label names is critical. 


Figure 5: Example for contextual label representation.


# E.3 Results

The results from various schemes of the new span is compared with TransferBERT and our model which encodes label names only. This is summarized in Table 10. 

TOKEN scheme is the simplest way to get a contextualized representation of a label where we pool the representations of all the tokens annotated with the label. Although performance of this scheme is better than TransferBERT, comparing with other schemes, we see that this model performs poorly. Here no new information is added to the model and the text that the label encoder and document encoder encodes is similar. In order to provide our model prior knowledge about the label name from BERT encoder, we use LABEL scheme. We see that this scheme performs better than TOKEN across datasets suggesting that the prior knowledge about label semantics helps to improve performance. 

One limitation with LABEL scheme is that the 

<table><tr><td></td><td></td><td>1 Shot</td><td>5 Shot</td><td>20 Shot</td><td>50 Shot</td></tr><tr><td rowspan="9">CoNLI03</td><td>TransferBERT</td><td>47.6 ±15.5</td><td>69.9 ±6.0</td><td>80.1 ±1.7</td><td>85.1 ±1.1</td></tr><tr><td>Ours, label name only</td><td>69.0 ±6.9</td><td>78.6 ±1.8</td><td>82.1 ±1.5</td><td>85.9 ±1.2</td></tr><tr><td>TOKEN</td><td>60.1 ±16.8</td><td>75.0 ±4.2</td><td>80.0 ±1.8</td><td>84.3 ±1.1</td></tr><tr><td>LABEL</td><td>61.4 ±12.7</td><td>74.2 ±2.9</td><td>80.4 ±1.9</td><td>84.6 ±1.2</td></tr><tr><td>[MASK]</td><td>61.2 ±6.1</td><td>72.9 ±5.8</td><td>81.5 ±2.2</td><td>85.3 ±0.9</td></tr><tr><td>BIO-TAG : [MASK]</td><td>60.8 ±15.4</td><td>74.5 ±5.6</td><td>81.3 ±1.5</td><td>85.2 ±0.8</td></tr><tr><td>(BIO-TAG) [MASK]</td><td>66.8 ±6.7</td><td>74.6 ±7.0</td><td>81.6 ±1.8</td><td>85.3 ±1.0</td></tr><tr><td>BIO-TAG : LABEL</td><td>69.2 ±6.4</td><td>76.1 ±2.1</td><td>80.8 ±1.9</td><td>84.9 ±1.1</td></tr><tr><td>(BIO-TAG) LABEL</td><td>70.8 ±4.2</td><td>76.5 ±1.6</td><td>81.2 ±2.0</td><td>84.7 ±1.1</td></tr><tr><td rowspan="9">WNUT17</td><td>TransferBERT</td><td>35.6 ±11.2</td><td>44.7 ±5.6</td><td>50.3 ±1.7</td><td>51.7 ±1.9</td></tr><tr><td>Ours, label name only</td><td>48.3 ±1.7</td><td>51.2 ±1.4</td><td>53.2 ±1.1</td><td>54.1 ±1.3</td></tr><tr><td>TOKEN</td><td>42.8 ±12.3</td><td>49.9 ±1.9</td><td>53.1 ±1.8</td><td>53.9 ±1.8</td></tr><tr><td>LABEL</td><td>48.9 ±3.0</td><td>51.4 ±2.1</td><td>53.0 ±1.6</td><td>53.9 ±1.5</td></tr><tr><td>[MASK]</td><td>45.0 ±3.5</td><td>47.1 ±2.2</td><td>50.2 ±2.3</td><td>51.9 ±1.6</td></tr><tr><td>BIO-TAG : [MASK]</td><td>46.8 ±2.8</td><td>49.6 ±1.7</td><td>51.3 ±2.8</td><td>52.7 ±1.0</td></tr><tr><td>(BIO-TAG) [MASK]</td><td>45.6 ±4.8</td><td>48.5 ±2.6</td><td>51.2 ±2.7</td><td>52.6 ±1.7</td></tr><tr><td>BIO-TAG : LABEL</td><td>51.2 ±2.2</td><td>52.6 ±1.8</td><td>53.6 ±1.4</td><td>54.8 ±0.6</td></tr><tr><td>(BIO-TAG) LABEL</td><td>51.9 ±1.8</td><td>52.3 ±1.2</td><td>53.7 ±1.5</td><td>54.0 ±1.3</td></tr><tr><td rowspan="9">NCBI-diseases</td><td>TransferBERT</td><td>15.1 ±9.4</td><td>19.5 ±6.0</td><td>37.0 ±4.1</td><td>51.2 ±4.1</td></tr><tr><td>Ours, label name only</td><td>31.4 ±9.2</td><td>30.2 ±4.3</td><td>45.8 ±3.4</td><td>57.3 ±2.6</td></tr><tr><td>TOKEN</td><td>18.7 ±10.3</td><td>22.5 ±6.4</td><td>40.9 ±5.6</td><td>53.8 ±4.1</td></tr><tr><td>LABEL</td><td>26.9 ±8.3</td><td>28.7 ±4.2</td><td>40.2 ±3.7</td><td>52.3 ±2.9</td></tr><tr><td>[MASK]</td><td>18.1 ±9.6</td><td>22.2 ±4.0</td><td>38.2 ±5.3</td><td>53.0 ±4.0</td></tr><tr><td>BIO-TAG : [MASK]</td><td>17.7 ±10.0</td><td>22.3 ±4.2</td><td>40.0 ±4.5</td><td>52.1 ±3.7</td></tr><tr><td>(BIO-TAG) [MASK]</td><td>17.5 ±11.5</td><td>23.6 ±4.1</td><td>38.8 ±4.7</td><td>51.9 ±4.0</td></tr><tr><td>BIO-TAG : LABEL</td><td>26.8 ±7.4</td><td>26.2 ±3.8</td><td>42.0 ±4.1</td><td>54.4 ±3.4</td></tr><tr><td>(BIO-TAG) LABEL</td><td>26.8 ±9.2</td><td>26.7 ±3.3</td><td>43.9 ±3.8</td><td>54.6 ±3.3</td></tr><tr><td rowspan="9">JNLPBA</td><td>TransferBERT</td><td>26.3 ±8.0</td><td>41.8 ±3.0</td><td>55.9 ±3.5</td><td>64.3 ±1.3</td></tr><tr><td>Ours, label name only</td><td>31.5 ±3.0</td><td>43.3 ±2.8</td><td>55.8 ±3.4</td><td>63.6 ±1.0</td></tr><tr><td>TOKEN</td><td>29.0 ±6.5</td><td>43.2 ±2.4</td><td>55.9 ±3.6</td><td>63.8 ±1.2</td></tr><tr><td>LABEL</td><td>28.4 ±4.3</td><td>40.8 ±2.5</td><td>54.3 ±3.4</td><td>62.5 ±1.3</td></tr><tr><td>[MASK]</td><td>25.4 ±6.5</td><td>36.5 ±2.2</td><td>51.0 ±3.7</td><td>60.2 ±1.5</td></tr><tr><td>BIO-TAG : [MASK]</td><td>24.9 ±5.1</td><td>36.0 ±2.5</td><td>50.5 ±4.2</td><td>60.5 ±1.7</td></tr><tr><td>(BIO-TAG) [MASK]</td><td>24.8 ±6.5</td><td>37.1 ±2.9</td><td>50.4 ±4.1</td><td>60.3 ±1.7</td></tr><tr><td>BIO-TAG : LABEL</td><td>30.4 ±4.6</td><td>41.9 ±2.5</td><td>55.5 ±3.3</td><td>62.9 ±1.1</td></tr><tr><td>(BIO-TAG) LABEL</td><td>30.1 ±3.2</td><td>41.4 ±2.2</td><td>55.1 ±3.2</td><td>62.8 ±1.5</td></tr><tr><td rowspan="3">FN-Person</td><td>TransferBERT</td><td>13.2 ±5.0</td><td>24.0 ±7.4</td><td>48.7 ±3.4</td><td>66.9 ±3.0</td></tr><tr><td>Ours, label name only</td><td>32.5 ±8.1</td><td>51.0 ±7.0</td><td>66.2 ±2.0</td><td>72.0 ±0.7</td></tr><tr><td>(BIO-TAG) LABEL</td><td>29.0 ±7.2</td><td>50.6 ±6.3</td><td>66.2 ±2.0</td><td>71.2 ±0.9</td></tr><tr><td rowspan="3">FN-Art</td><td>TransferBERT</td><td>19.4 ±10.9</td><td>43.1 ±9.8</td><td>69.5 ±1.7</td><td>98.9 ±0.3</td></tr><tr><td>Ours, label name only</td><td>44.5 ±8.8</td><td>56.3 ±4.6</td><td>70.5 ±1.8</td><td>99.1 ±0.1</td></tr><tr><td>(BIO-TAG) LABEL</td><td>41.3 ±10.8</td><td>56.0 ±3.8</td><td>69.4 ±2.0</td><td>98.9 ±0.2</td></tr></table>


Table 10: Results on development set across all datasets. FN-Person = FEW-NERD-Person. FN-Art = FEW-NERD-Art. All numbers indicate micro F1 scores and are average of 10 runs with different support set sampling. 
