# Robustness-Congruent Adversarial Training for Secure Machine Learning Model Updates  :arrows_counterclockwise: :shield:

Machine-learning models demand periodic updates to improve their average accuracy, exploiting novel architectures and additional data. However, a newly updated model may commit mistakes the previous model did not make. Such misclassifications are referred to as *negative flips*, experienced by users as a regression of performance. 
In this work, we show that this problem also affects robustness to adversarial examples,  hindering the development of secure model update practices. In particular, when updating a model to improve its adversarial robustness, previously ineffective adversarial attacks on some inputs may become successful, causing a regression in the perceived security of the system.
We propose a novel technique, named robustness-congruent adversarial training, to address this issue. It amounts to fine-tuning a model with adversarial training, while constraining it to retain higher robustness on the samples for which no adversarial example was found before the update. We show that our algorithm and, more generally, learning with non-regression constraints, provides a theoretically-grounded framework to train consistent estimators. 
Our experiments on robust models for computer vision confirm that both accuracy and robustness, even if improved after model update, can be affected by negative flips, and our robustness-congruent adversarial training can mitigate the problem, outperforming competing baseline methods.

📌 Just Accepted at IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2025  
📄 [preprint](https://arxiv.org/abs/2402.17390)
