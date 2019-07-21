#!/usr/bin/env bash
cd ./submission
python predict.py --path-images-csv ../data/check_submission_data_v2/check_submission_data/check_images.csv --path-test-dir ../data/check_submission_data_v2/check_submission_data/check --path-submission-csv ../test_submission.csv