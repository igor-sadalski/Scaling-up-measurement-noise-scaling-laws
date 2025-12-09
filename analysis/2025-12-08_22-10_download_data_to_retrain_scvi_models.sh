#!/bin/bash
export BASE_FOLDER="/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/outputs/2025-12-08_22-10_download_data_to_retrain_scvi_models"

cp -r /home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/outputs/2025-12-07_23-21_evaluate_all_shendure_geneformer_models_on_mid_quality_data/* $BASE_FOLDER/

aws s3 cp --recursive s3://somite-share/igor/noise-loss/data/shendure/validation/1.0/ $BASE_FOLDER/shendure/validation/1.0/

ACCURACIES=(1.0 0.5414548 0.2931733 0.1587401 0.0859506 0.0465384 0.0251984 0.0136438 0.0073875 0.004)

for accuracy in "${ACCURACIES[@]}"; do
    if [ "$accuracy" != "1.0" ]; then
		echo "Copying test set for accuracy: $accuracy"
		cp -r $BASE_FOLDER/shendure/validation/1.0 $BASE_FOLDER/shendure/validation/$accuracy
    fi
done

# only data for model where we dont have scvi
# aws s3 cp --recursive s3://somite-share/igor/noise-loss/data/shendure/10000000/1.0/preprocessed/ $BASE_FOLDER/shendure/10000000/1.0/preprocessed/
aws s3 cp --recursive s3://somite-share/igor/noise-loss/data/shendure/10000000/0.5414548/preprocessed/ $BASE_FOLDER/shendure/10000000/0.5414548/preprocessed/
aws s3 cp --recursive s3://somite-share/igor/noise-loss/data/shendure/10000000/0.0251984/preprocessed/ $BASE_FOLDER/shendure/10000000/0.0251984/preprocessed/

find $BASE_FOLDER/shendure/10000000 -type d -name '*checkpoint*' -exec rm -rf {} +

