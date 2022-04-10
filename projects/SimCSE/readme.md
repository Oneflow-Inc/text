# SimCSE
Reproduce Supervised-simcse and Unsupervised-simcse with OneFlow.

## Introduce
SimCSE is a sentence representation learning method, in which there are two training methods: supervised learning and unsupervised learning. The unsupervised learning method is to input sentences and predict itself in the comparison target, and only use the standard dropout as noise; The supervised learning method uses the NLI data set, taking 'entry' as a positive sample and 'contrast' as a negative sample for supervised learning. This task uses Spearman to evaluate the model's performance on STS dataset, and uses Alignment and Uniformity to measure the effect of contrastive learning. 
- 《SimCSE: Simple Contrastive Learning of Sentence Embeddings》: https://arxiv.org/pdf/2104.08821.pdf
- Official GitHub: https://github.com/princeton-nlp/SimCSE

## Evaluation
Dataset: SNLI+STS
|      Unsupervised-Model        |STS-B dev |STS-B test|Pool type |
|:-------------------------------|:--------:|:--------:|:--------:|
|unsup-simcse-bert-base-chinese  |73.25  | 66.58   |    cls   |
|unsup-simcse-bert-base-chinese  |73.08  | 66.43   | last-avg |
|unsup-simcse-bert-base-chinese  |67.98  | 64.00   |pooled    |
|unsup-simcse-bert-base-chinese  |73.67  | 67.82   |first-last-avg|



Dataset: SNLI
|       Supervised-Model         |STS-B dev |STS-B test|Pool type |
|:-------------------------------|:--------:|:--------:|:--------:|
|unsup-simcse-bert-base-chinese  |80.39  |77.54   |    cls   |
|unsup-simcse-bert-base-chinese  |80.14  |76.50   | last-avg |
|unsup-simcse-bert-base-chinese  |77.21  |73.14   |pooled    |
|unsup-simcse-bert-base-chinese  |79.82  |74.72   |first-last-avg|