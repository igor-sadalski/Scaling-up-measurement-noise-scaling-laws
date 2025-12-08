BASE_FOLDER="/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/outputs/2025-12-07_23-12_download_different_geneformer_models"

#download models trained on medium quality data (e.g. PBMC 100000 10% original umi counts) and run them on test set with full data quality

#shendure
# Download test data and utils (only once)
aws s3 cp --recursive s3://measurement-noise-scaling-laws/data/shendure/test/1.0/ $BASE_FOLDER/shendure/test/1.0/
aws s3 cp --recursive s3://measurement-noise-scaling-laws/data/shendure/utils/ $BASE_FOLDER/shendure/utils/

# Accuracies to download
ACCURACIES=(1.0 0.5414548 0.2931733 0.1587401 0.0859506 0.0465384 0.0251984 0.0136438 0.0073875 0.004)

aws s3 cp --recursive s3://measurement-noise-scaling-laws/data/shendure/10000000/0.0859506/results/ $BASE_FOLDER/shendure/10000000/0.0859506/results/

# Download results for each accuracy
for accuracy in "${ACCURACIES[@]}"; do
    # echo "Downloading results for accuracy: $accuracy"
    aws s3 cp --recursive s3://measurement-noise-scaling-laws/data/shendure/10000000/$accuracy/results/ $BASE_FOLDER/shendure/10000000/$accuracy/results/
	#this way we will evaluate all the models on the same test set of high quality
    if [ "$accuracy" != "1.0" ]; then
		echo "Copying test set for accuracy: $accuracy"
        cp -r $BASE_FOLDER/shendure/test/1.0 $BASE_FOLDER/shendure/test/$accuracy
    fi
done