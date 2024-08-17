In the field of image segmentation, there are often multiple valid solutions for a given input. In medical imaging, experts frequently disagree on precise object boundaries. Our focus is on the quantification of this inherent uncertainty during glioma segmentation on
MRI brain scans, which helps prioritize areas for manual review by experts and improves clinical decisionmaking.
Subsequently, as our primary goal, we are setting the precise identification of tumors, which includes assessing the segmentation accuracy with IoU and loss function, and uncertainties quantification with entropy based on ensemble predictions.
Our approach for epistemic uncertainty is to use ensembles of UNets, which produce pixelwise estimates. For aleatoric uncertainty we will use test-time augmentation.
