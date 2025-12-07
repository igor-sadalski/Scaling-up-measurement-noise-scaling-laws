BASE_FOLDER="/home/igor/igor_repos/noise_scaling_laws/Scaling-up-measurement-noise-scaling-laws/analysis/outputs/2025-12-02_19_48_download_necessary_models"


#download models trained on medium quality data (e.g. PBMC 100000 10% original umi counts) and run them on test set with full data quality

#PBMC
aws s3 cp --recursive s3://measurement-noise-scaling-laws/data/PBMC/100000/0.1072766/results/ $BASE_FOLDER/PBMC/100000/0.1072766/results/
aws s3 cp --recursive s3://measurement-noise-scaling-laws/data/PBMC/test/1.0/ $BASE_FOLDER/PBMC/test/1.0/
aws s3 cp --recursive s3://measurement-noise-scaling-laws/data/PBMC/utils/ $BASE_FOLDER/PBMC/utils/
aws s3 cp --recursive s3://somite-share/igor/noise-loss/data/PBMC/100000/0.1072766/preprocessed/ $BASE_FOLDER/PBMC/100000/0.1072766/preprocessed/
#simple rename so we can use our current framework for consistency
mv $BASE_FOLDER/PBMC/test/1.0 $BASE_FOLDER/PBMC/test/0.1072766 

#larry
aws s3 cp --recursive s3://somite-share/igor/noise-loss/data/larry/100000/0.0847557/results/ $BASE_FOLDER/larry/100000/0.0847557/results/
aws s3 cp --recursive s3://measurement-noise-scaling-laws/data/larry/test/1.0/ $BASE_FOLDER/larry/test/1.0/
aws s3 cp --recursive s3://measurement-noise-scaling-laws/data/larry/utils/ $BASE_FOLDER/larry/utils/
mv $BASE_FOLDER/larry/test/1.0 $BASE_FOLDER/larry/test/0.0847557

#shendure
aws s3 cp --recursive s3://measurement-noise-scaling-laws/data/shendure/10000000/0.0859506/results/ $BASE_FOLDER/shendure/10000000/0.0859506/results/
aws s3 cp --recursive s3://measurement-noise-scaling-laws/data/shendure/test/1.0/ $BASE_FOLDER/shendure/test/1.0/
aws s3 cp --recursive s3://measurement-noise-scaling-laws/data/shendure/utils/ $BASE_FOLDER/shendure/utils/
mv $BASE_FOLDER/shendure/test/1.0 $BASE_FOLDER/shendure/test/0.0859506

#merfish
aws s3 cp --recursive s3://measurement-noise-scaling-laws/data/merfish/60000/0.0905502/results/ $BASE_FOLDER/merfish/60000/0.0905502/results/
aws s3 cp --recursive s3://measurement-noise-scaling-laws/data/merfish/test/1.0/ $BASE_FOLDER/merfish/test/1.0/
aws s3 cp --recursive s3://measurement-noise-scaling-laws/data/merfish/utils/ $BASE_FOLDER/merfish/utils/
mv $BASE_FOLDER/merfish/test/1.0 $BASE_FOLDER/merfish/test/0.0905502
