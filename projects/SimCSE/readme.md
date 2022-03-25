# SimCSE
Reproduce Supervised-simcse and Unsupervised-simcse with OneFlow.

## introduce
SimCSE is a sentence representation learning method, in which there are two training methods: supervised learning and unsupervised learning. The unsupervised learning method is to input sentences and predict itself in the comparison target, and only use the standard dropout as noise; The supervised learning method uses the NLI data set, taking 'entry' as a positive sample and 'contrast' as a negative sample for supervised learning.
- Paper: https://arxiv.org/pdf/2104.08821.pdf
- Official GitHub: https://github.com/princeton-nlp/SimCSE

## Evaluation
Dataset: SNLI+STS
|      Unsupervised-Model        |STS-B dev |
|:-------------------------------|:--------:|
|unsup-simcse-bert-base-chinese  |   72.55  |

Dataset: SNLI
|       Supervised-Model         |STS-B dev |
|:-------------------------------|:--------:|
|unsup-simcse-bert-base-chinese  |   80.23  |