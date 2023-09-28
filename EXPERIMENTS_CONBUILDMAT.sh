#!/usr/bin/env bash

DATASET_ROOT=./SCCDNet-corrected
RESULTS=./RESULTS

TRAIN_ARGS=(--LEARNING_RATE=0.001
            --AUGMENTATION=True
            --BATCH_SIZE=10
            --BCE_LOSS_W=True
            --BEST_MODEL_TYPE=both
            --DELTA_CLS_LOSS=0.1
            --DILATE=1
            --DYN_BALANCED_LOSS=False
            --EPOCHS=200
            --NUM_SEGMENTED=6164
            --FREQUENCY_SAMPLING=False
            --GPU=0
            --GRADIENT_ADJUSTMENT=True
            --DATASET=sccdnet
            --DATASET_PATH="${DATASET_ROOT}"
            --ON_DEMAND_READ=False
            --PXL_DISTANCE=2
            --RESULTS_PATH="${RESULTS}"
            --SAVE_IMAGES=True
            --SEG_BLACK=False
            --USE_BEST_MODEL=True
            --VALIDATE=True
            --VALIDATE_ON_TEST=True
            --VALIDATION_N_EPOCHS=2
            --WEIGHTED_SEG_LOSS=False)

mkdir -p ./RESULTS/sccdnet/

# shellcheck disable=SC2068
CUDA_VISIBLE_DEVICES=0 python -u train_net.py --RUN_NAME=segdecnet++_run_1 --REPRODUCIBLE_RUN=101 ${TRAIN_ARGS[@]} > $RESULTS/sccdnet/paper_sccdnet_1.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u train_net.py --RUN_NAME=segdecnet++_run_2 --REPRODUCIBLE_RUN=102 ${TRAIN_ARGS[@]} > $RESULTS/sccdnet/paper_sccdnet_2.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u train_net.py --RUN_NAME=segdecnet++_run_3 --REPRODUCIBLE_RUN=103 ${TRAIN_ARGS[@]} > $RESULTS/sccdnet/paper_sccdnet_3.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u train_net.py --RUN_NAME=segdecnet++_run_4 --REPRODUCIBLE_RUN=104 ${TRAIN_ARGS[@]} > $RESULTS/sccdnet/paper_sccdnet_4.out 2>&1 &
CUDA_VISIBLE_DEVICES=4 python -u train_net.py --RUN_NAME=segdecnet++_run_5 --REPRODUCIBLE_RUN=105 ${TRAIN_ARGS[@]} > $RESULTS/sccdnet/paper_sccdnet_5.out 2>&1 &
wait

########################################################################################################################################
# evaluate the best model based on stored output

echo "SegDecNet++ for Segmentation (segdecnet++_run_1/seg_pred_bin)"
python evaluate_output.py -ground_truth_dir $DATASET_ROOT/test/masks -pred_dir $RESULTS/sccdnet/segdecnet++_run_1/seg_pred_bin -threshold 0.5

echo "SegDecNet++ for Segmentation (segdecnet++_run_2/seg_pred_bin)"
python evaluate_output.py -ground_truth_dir $DATASET_ROOT/test/masks -pred_dir $RESULTS/sccdnet/segdecnet++_run_2/seg_pred_bin -threshold 0.5

echo "SegDecNet++ for Segmentation (segdecnet++_run_3/seg_pred_bin)"
python evaluate_output.py -ground_truth_dir $DATASET_ROOT/test/masks -pred_dir $RESULTS/sccdnet/segdecnet++_run_3/seg_pred_bin -threshold 0.5

echo "SegDecNet++ for Segmentation (segdecnet++_run_4/seg_pred_bin)"
python evaluate_output.py -ground_truth_dir $DATASET_ROOT/test/masks -pred_dir $RESULTS/sccdnet/segdecnet++_run_4/seg_pred_bin -threshold 0.5

echo "SegDecNet++ for Segmentation (segdecnet++_run_5/seg_pred_bin)"
python evaluate_output.py -ground_truth_dir $DATASET_ROOT/test/masks -pred_dir $RESULTS/sccdnet/segdecnet++_run_5/seg_pred_bin -threshold 0.5

########################################################################################################################################
# evaluate by dataset source

for subset in CFD cracktree200 DeepCrack forest GAPS rissbilder noncrack; do
  echo "########################################################## $subset ##########################################################"

  echo "SegDecNet for Segmentation (segdecnet++_run_1/seg_pred_bin)"
  python evaluate_output.py -ground_truth_dir $DATASET_ROOT/test/masks -gt_filenames ./splits/SCCDNet/test_$subset.txt -pred_dir $RESULTS/sccdnet/segdecnet++_run_1/seg_pred_bin -threshold 0.5

  echo "SegDecNet for Segmentation (segdecnet++_run_2/seg_pred_bin)"
  python evaluate_output.py -ground_truth_dir $DATASET_ROOT/test/masks -gt_filenames ./splits/SCCDNet/test_$subset.txt -pred_dir $RESULTS/sccdnet/segdecnet++_run_2/seg_pred_bin -threshold 0.5

  echo "SegDecNet for Segmentation (segdecnet++_run_3/seg_pred_bin)"
  python evaluate_output.py -ground_truth_dir $DATASET_ROOT/test/masks -gt_filenames ./splits/SCCDNet/test_$subset.txt -pred_dir $RESULTS/sccdnet/segdecnet++_run_3/seg_pred_bin -threshold 0.5

  echo "SegDecNet for Segmentation (segdecnet++_run_4/seg_pred_bin)"
  python evaluate_output.py -ground_truth_dir $DATASET_ROOT/test/masks -gt_filenames ./splits/SCCDNet/test_$subset.txt -pred_dir $RESULTS/sccdnet/segdecnet++_run_4/seg_pred_bin -threshold 0.5

  echo "SegDecNet for Segmentation (segdecnet++_run_5/seg_pred_bin)"
  python evaluate_output.py -ground_truth_dir $DATASET_ROOT/test/masks -gt_filenames ./splits/SCCDNet/test_$subset.txt -pred_dir $RESULTS/sccdnet/segdecnet++_run_5/seg_pred_bin -threshold 0.5

done
