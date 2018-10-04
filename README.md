# Temporal Saliency Adaptation in Egocentric Videos

Our work is based on SalGAN, a computational model of saliency to predict human fixations on still images. In terms of architecture, we have added a convolutional LSTM layer on top of the frame-based saliency predictions.
SalGAN can be found in the static salgan folder, its original source is https://github.com/imatge-upc/saliency-salgan-2017
We used the salience metrics from a python implementation https://github.com/tarunsharma1/saliency_metrics

You may find more information regarding our work here: https://imatge-upc.github.io/saliency-2018-videosalgan/
