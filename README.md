# Cross-Encoder
Cross-Encoder for Unsupervised Gaze Representation Learning, published in ICCV 2021.
paper link: https://openaccess.thecvf.com/content/ICCV2021/papers/Sun_Cross-Encoder_for_Unsupervised_Gaze_Representation_Learning_ICCV_2021_paper.pdf

In order to train 3D gaze estimators without too many annotations,we propose an unsupervised learning framework, Cross-Encoder, to leverage the unlabeled data to learn suitable representation for gaze estimation. 
To address the issue that the feature of gaze is always intertwined with the appearance of the eye, Cross-Encoder disentangles the features using a latent-code-swapping mechanism on eye-consistent image pairs and gaze-similar ones.
Specifically, each image is encoded as a gaze feature and an eye feature.
Cross-Encoder is trained to reconstruct each image in the eye-consistent pair according to its gaze feature and the other's eye feature, but to reconstruct each image in the gaze-similar pair according to its eye feature and the other's gaze feature.
Experimental results show the validity of our work.
First, using the Cross-Encoder-learned gaze representation, the gaze estimator trained with very few samples outperforms the ones using other unsupervised learning methods, under both within-dataset and cross-dataset protocol. 
Second, ResNet18 pretrained by Cross-Encoder is competitive with state-of-the-art gaze estimation methods.
Third, ablation study shows that Cross-Encoder disentangles the gaze feature and eye feature.

utils/checkpoints_manager.py and models/densenet.py are from https://github.com/NVlabs/few_shot_gaze.git
