https://github.com/bagxi/idrnd-anti-spoofing-challenge-solution

# [IDRND Anti-spoofing Challenge](https://datasouls.com/c/idrnd-antispoof/description) Solution

## NN pipeline

	# prepare data & train 5 folds
	python run_nn.py prepare-folds \
		--in-dir=./data/train \
		--out-csv=./data/train/dataset.csv \
		--holdout-csv=./data/train/holdout.csv \
		--n-folds=5 \
		--holdout-size=0.2
	bash run.sh --dataset train \
		--model resnet18 \
		--n-epochs 30 \
		--batch-size 256 \
		--n-workers 4 \
		--fast
	python run_nn.py distil-model \
		--model=resnet18 \
		--in-weights=./models/easy_gold.pth \
		--out-weights=./models/easy_gold.pth

	# infer
	python run_nn.py infer \
		--in-csv=$PATH_INPUT/meta.csv \
		--in-dir=$PATH_INPUT \
		--out-csv=$PATH_OUTPUT/solution.csv \
		--model=resnet18 \
		--weights-path=./models/easy_gold.pth \
		--batch-size=256 \
		--n-workers=4

## LBP pipeline

	# prepare data & train model
	python run_lbp.py prepare-cutout-datasets \
		--in-dir=./data/train \
		--out-dir-crops=./data/crops \
		--out-dir-cutout=./data/cutout \
		--verbose=True
	python run_lbp.py prepare-lbp-dataset \
		--dirpath=./data/crops \
		--features-npy=./data/crops/features.npy \
		--targets-csv=./data/crops/dataset.csv \
		--verbose=True
	python run_lbp.py train \
		--features-npy=./data/crops/features.npy \
		--targets-csv=./data/crops/dataset.csv \
		--n-splits=5 \
		--n-repeats=10 \
		--logdir=./logs/lbp

	# infer
	python run_lbp.py infer \
		--in-csv=$PATH_INPUT/meta.csv \
		--in-dir=$PATH_INPUT \
		--out-csv=$PATH_OUTPUT/solution.csv \
		--weights-path=./logs/lbp/model.pkl

## Copyright and License
© Copyright 2019-present Yauheni Kachan.
Licensed under the [MIT License](LICENSE.md).
