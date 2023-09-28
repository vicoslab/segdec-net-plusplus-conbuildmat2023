import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import SegDecNetPlusPlus, SegDecNetOriginalJIM
import numpy as np
import os
from torch import nn as nn
import torch
import utils
import pandas as pd
from data.dataset_catalog import get_dataset
import random
import cv2
from config import Config
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from timeit import default_timer as timer
from datetime import timedelta
from torchinfo import summary
from tqdm import tqdm
LVL_ERROR = 10
LVL_INFO = 5
LVL_DEBUG = 1

LOG = 1  # Will log all mesages with lvl greater than this
SAVE_LOG = True

WRITE_TENSORBOARD = False


class End2End:
    def __init__(self, cfg: Config):
        self.cfg: Config = cfg
        self.storage_path: str = os.path.join(self.cfg.RESULTS_PATH, self.cfg.DATASET)

    def _log(self, message, lvl=LVL_INFO):
        time = datetime.now().strftime("%d-%m-%y %H:%M")
        n_msg = f"{time} {self.run_name} {message}"
        if lvl >= LOG:
            print(n_msg)

    def train(self):
        self._set_results_path()
        self._create_results_dirs()
        self.print_run_params()
        self.set_seed()

        device = self._get_device()
        model = self._get_model().to(device)
        optimizer = self._get_optimizer(model)
        scheduler = self._get_scheduler(optimizer)

        # Save current learning method to model's directory
        utils.save_current_learning_method(save_path=self.run_path)

        train_loader = get_dataset("TRAIN", self.cfg)
        validation_loader = get_dataset("VAL", self.cfg)

        loss_seg, loss_dec = self._get_loss(is_seg=True, pos_weight=train_loader.dataset.pos_weight_seg), self._get_loss(is_seg=False, pos_weight=train_loader.dataset.pos_weight_dec)

        tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path) if WRITE_TENSORBOARD else None

        train_start = timer()
        losses, validation_data, best_model_metrics, validation_metrics, lrs, difficulty_score_dict = self._train_model(device, model, train_loader, loss_seg, loss_dec, optimizer, scheduler, validation_loader, tensorboard_writer)
        end = timer()
        self._log(f"Training time: {timedelta(seconds=end-train_start)}")
        train_results = (losses, validation_data, validation_metrics, lrs)
        self._save_train_results(train_results)
        self._save_model(model)

        # Save difficulty_score_dict
        np.save(os.path.join(self.run_path, "difficulty_score_dict.npy"), difficulty_score_dict)

        self.eval(model=model, device=device, save_images=self.cfg.SAVE_IMAGES, plot_seg=False, reload_final=False, best_model_metrics=best_model_metrics)

        self._save_params()

        # Print model's trainable parameters # and save model's summary to file
        self._log(f"Model's trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print(summary(model, input_size=torch.Size([self.cfg.BATCH_SIZE, self.cfg.INPUT_CHANNELS, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_WIDTH]), verbose=0), file=open(os.path.join(self.run_path, "model_summary.txt"), 'w', encoding="utf-8"))

    def eval(self, model, device, save_images, plot_seg, reload_final, eval_loader=None, best_model_metrics=None):
        self.reload_model(model, reload_final)
        is_validation = True
        if eval_loader is None:
            eval_loader = get_dataset("TEST", self.cfg)
            is_validation = False
        eval_start = timer()
        self.eval_model(device, model, eval_loader, save_folder=self.outputs_path, save_images=save_images, is_validation=is_validation, plot_seg=plot_seg, thresholds=best_model_metrics)
        end = timer()
        self._log(f"Evaluation time: {timedelta(seconds=end-eval_start)}")

    def training_iteration(self, data, device, model, criterion_seg, criterion_dec, optimizer, weight_loss_seg, weight_loss_dec,
                           tensorboard_writer, iter_index):
        images, seg_masks, is_segmented, sample_names, is_pos, _ = data

        batch_size = self.cfg.BATCH_SIZE
        memory_fit = self.cfg.MEMORY_FIT  # Not supported yet for >1

        num_subiters = int(batch_size / memory_fit)

        total_loss = 0
        total_correct = 0

        optimizer.zero_grad()

        total_loss_seg = 0
        total_loss_dec = 0

        difficulty_score = np.zeros(batch_size)

        for sub_iter in range(num_subiters):
            images_ = images[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            seg_mask_ = seg_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            is_pos_ = seg_mask_.max().reshape((memory_fit, 1)).to(device)

            if tensorboard_writer is not None and iter_index % 100 == 0:
                tensorboard_writer.add_image(f"{iter_index}/image", images_[0, :, :, :])

            decision, seg_mask_predicted = model(images_)

            if is_segmented[sub_iter]:
                loss_seg = criterion_seg(seg_mask_predicted, seg_mask_)
                loss_dec = criterion_dec(decision, is_pos_)

                if self.cfg.HARD_NEG_MINING is not None:
                    _, _, difficulty_score_mode = self.cfg.HARD_NEG_MINING
                    if difficulty_score_mode == 1:
                        difficulty_score[sub_iter] = loss_seg.item()
                    elif difficulty_score_mode == 2:
                        threshold = 0.5
                        y_true = seg_mask_.detach().cpu().numpy()[0][0].astype(np.uint8)
                        y_pred = (seg_mask_predicted.detach().cpu().numpy()[0][0]>threshold).astype(np.uint8)

                        fp = sum(sum((y_true==0)&(y_pred==1))).item()
                        fn = sum(sum((y_true==1)&(y_pred==0))).item()

                        difficulty_score[sub_iter] = loss_seg.item() * ((2 * fp) + fn + 1)
                    elif difficulty_score_mode == 3:
                        seg_mask_predicted = nn.Sigmoid()(seg_mask_predicted)
                        seg_mask_predicted_max = seg_mask_predicted.detach().cpu().numpy()[0][0].max()
                        classification = nn.Sigmoid()(decision).item()
                        difficulty_score[sub_iter] = abs(seg_mask_predicted_max - classification)


                total_loss_seg += loss_seg.item()
                total_loss_dec += loss_dec.item()

                total_correct += (decision > 0.0).item() == is_pos_.item()
                loss = weight_loss_seg * loss_seg + weight_loss_dec * loss_dec
            else:
                loss_dec = criterion_dec(decision, is_pos_)
                total_loss_dec += loss_dec.item()

                total_correct += (decision > 0.0).item() == is_pos_.item()
                loss = weight_loss_dec * loss_dec

            total_loss += loss.item()

            loss.backward()

        # Backward and optimize
        optimizer.step()
        optimizer.zero_grad()

        return total_loss_seg, total_loss_dec, total_loss, total_correct, difficulty_score

    def _train_model(self, device, model, train_loader, criterion_seg, criterion_dec, optimizer, scheduler, validation_set, tensorboard_writer):
        losses = []
        validation_data = []
        validation_metrics = []
        lrs = []
        max_validation = -1
        max_f_measure = -1
        best_dice = -1
        best_f1 = -1
        validation_step = self.cfg.VALIDATION_N_EPOCHS

        num_epochs = self.cfg.EPOCHS
        samples_per_epoch = len(train_loader) * self.cfg.BATCH_SIZE

        difficulty_score_dict = dict()

        self.set_dec_gradient_multiplier(model, 0.0)

        for epoch in range(num_epochs):
            if epoch % 5 == 0:
                self._save_model(model, f"ep_{epoch:02}.pth")

            model.train()

            weight_loss_seg, weight_loss_dec = self.get_loss_weights(epoch)
            dec_gradient_multiplier = self.get_dec_gradient_multiplier()
            self.set_dec_gradient_multiplier(model, dec_gradient_multiplier)

            epoch_loss_seg, epoch_loss_dec, epoch_loss = 0, 0, 0
            epoch_correct = 0

            difficulty_score_dict[epoch] = []

            from timeit import default_timer as timer

            time_acc = 0
            start = timer()
            for iter_index, (data) in enumerate(tqdm(train_loader)):
                start_1 = timer()
                curr_loss_seg, curr_loss_dec, curr_loss, correct, difficulty_score = self.training_iteration(data, device, model,
                                                                                           criterion_seg,
                                                                                           criterion_dec,
                                                                                           optimizer, weight_loss_seg,
                                                                                           weight_loss_dec,
                                                                                           tensorboard_writer, (epoch * samples_per_epoch + iter_index))

                end_1 = timer()
                time_acc = time_acc + (end_1 - start_1)

                epoch_loss_seg += curr_loss_seg
                epoch_loss_dec += curr_loss_dec
                epoch_loss += curr_loss

                epoch_correct += correct

                if self.cfg.HARD_NEG_MINING is not None:
                    train_loader.batch_sampler.update_sample_loss_batch(data, difficulty_score, index_key=5)

                difficulty_score_dict[epoch].append({index.item(): round(score, 2) for index, score in zip(data[-1], difficulty_score)})

            end = timer()


            epoch_loss_seg = epoch_loss_seg / samples_per_epoch
            epoch_loss_dec = epoch_loss_dec / samples_per_epoch
            epoch_loss = epoch_loss / samples_per_epoch
            losses.append((epoch_loss_seg, epoch_loss_dec, epoch_loss, epoch))

            self._log(f"Epoch {epoch + 1}/{num_epochs} ==> avg_loss_seg={epoch_loss_seg:.5f}, avg_loss_dec={epoch_loss_dec:.5f}, avg_loss={epoch_loss:.5f}, correct={epoch_correct}/{samples_per_epoch}, in {end - start:.2f}s/epoch (fwd/bck in {time_acc:.2f}s/epoch)")

            if self.cfg.SCHEDULER is not None:
                scheduler.step()
                last_learning_rate = scheduler.get_last_lr()[-1]
                self._log(f"Last computing learning rate by scheduler: {last_learning_rate}")
                lrs.append((epoch, last_learning_rate))
            else:
                lrs.append((epoch, self._get_learning_rate(optimizer=optimizer)))

            self._log(f"Last computing learning rate by optimizer: {self._get_learning_rate(optimizer=optimizer)}")

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("Loss/Train/segmentation", epoch_loss_seg, epoch)
                tensorboard_writer.add_scalar("Loss/Train/classification", epoch_loss_dec, epoch)
                tensorboard_writer.add_scalar("Loss/Train/joined", epoch_loss, epoch)
                tensorboard_writer.add_scalar("Accuracy/Train/", epoch_correct / samples_per_epoch, epoch)

            if self.cfg.VALIDATE and (epoch % validation_step == 0 or epoch == num_epochs - 1):
                validation_ap, validation_accuracy, val_metrics = self.eval_model(device=device, model=model, eval_loader=validation_set, save_folder=None, save_images=False, is_validation=True, plot_seg=False)
                validation_data.append((validation_ap, epoch))
                validation_metrics.append((epoch, val_metrics))

                if val_metrics['Dice'] > best_dice:
                    best_dice = val_metrics['Dice']
                    best_seg_model_metrics = val_metrics
                    self._save_model(model, "best_seg_dict.pth")
                
                if val_metrics['best_f_measure'] > max_f_measure:
                    max_f_measure = val_metrics['best_f_measure']
                    best_dec_model_metrics = val_metrics
                    self._save_model(model, "best_dec_dict.pth")

                if val_metrics['Dice'] >= best_dice and val_metrics['best_f_measure'] >= max_f_measure:
                    best_dice = val_metrics['Dice']
                    max_f_measure = val_metrics['best_f_measure']
                    best_model_metrics = val_metrics
                    self._save_model(model, "best_state_dict.pth")

                model.train()
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("Accuracy/Validation/", validation_accuracy, epoch)

        if self.cfg.BEST_MODEL_TYPE == "dec":
            best_model_metrics = best_dec_model_metrics
        elif self.cfg.BEST_MODEL_TYPE == "seg":
            best_model_metrics = best_seg_model_metrics

        return losses, validation_data, best_model_metrics, validation_metrics, lrs, difficulty_score_dict

    def eval_model_speed(self, device, model, eval_loader):
        model.eval()

        cuda_time = []
        cuda_mem_usage = []
        cpu_time = []
        from itertools import chain

        iter = list(chain(eval_loader.dataset.neg_samples,eval_loader.dataset.pos_samples))

        from torchvision.transforms import functional as F

        N = 1000
        start = time.time()
        for index, data_point in enumerate(tqdm(iter)):
            _, _, _, image_path, _, _, _ = data_point
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = F.to_tensor(cv2.resize(image, dsize=(self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT))).unsqueeze(0)

            image = image.to(device)
            # ensure all work is done before next loop
            torch.cuda.synchronize()
            if index % 20 == 0 and False:
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             profile_memory=False, record_shapes=True, with_flops=True) as prof:
                    prediction, seg_mask_predicted = model(image)
                    # ensure all work is done before next loop
                    torch.cuda.synchronize()

                cuda_events = [n for n in prof.key_averages() if n.device_type == torch.autograd.DeviceType.CUDA]
                cuda_time.append(sum([n.self_cuda_time_total for n in cuda_events]))
                cpu_time.append(sum([n.self_cpu_time_total for n in prof.key_averages()]))
                cuda_mem_usage.append(sum([n.cuda_memory_usage for n in prof.key_averages()]))
                print(cuda_time[-1])
                print(cpu_time[-1])
            else:
                prediction, seg_mask_predicted = model(image)

            prediction = torch.sigmoid(prediction)
            seg_mask_predicted = torch.sigmoid(seg_mask_predicted)

            # ensure all work is done before next loop
            torch.cuda.synchronize()

        end = time.time()
        time_python_total = end-start
        time_python_per_img_ms = time_python_total*1000/N

        print(f'total python time: {time_python_total} sec, per image: {time_python_per_img_ms} ms')
        print(f'total python FPS: {1000.0/time_python_per_img_ms}')
        if len(cuda_time) > 0:
            print('avg cuda time: ', np.mean(cuda_time)/1000, ' ms')
        if len(cuda_mem_usage) > 0:
            print('avg cuda mem usage: ', np.mean(cuda_mem_usage) / 10**6, ' MB')
        print("num_params:", sum([p.numel() for p in model.parameters() if p.requires_grad]) / 10 ** 6)

    def eval_model(self, device, model, eval_loader, save_folder, save_images, is_validation, plot_seg, thresholds=None):
        model.eval()

        dsize = self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT

        res = []
        predictions, predictions_truths = [], []

        predicted_segs, true_segs = [], []
        samples = {"images": list(), "image_names": list()}

        for iii, data_point in enumerate(tqdm(eval_loader)):
            image, seg_mask, _, sample_name, is_pos, _ = data_point
            image, seg_mask = image.to(device), seg_mask.to(device)
            is_pos = is_pos.item()
            prediction, seg_mask_predicted = model(image)

            prediction = nn.Sigmoid()(prediction)
            seg_mask_predicted = nn.Sigmoid()(seg_mask_predicted)

            prediction = prediction.item()
            image = image.detach().cpu().numpy()
            seg_mask = seg_mask.detach().cpu().numpy()
            seg_mask_predicted = seg_mask_predicted.detach().cpu().numpy()

            image = cv2.resize(np.transpose(image[0, :, :, :], (1, 2, 0)), dsize)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            predictions.append(prediction)
            predictions_truths.append(is_pos)
            res.append((prediction, None, None, is_pos, sample_name[0]))

            seg_mask_predicted = seg_mask_predicted[0][0]
            seg_mask = seg_mask[0][0]
            predicted_segs.append(seg_mask_predicted)
            samples["image_names"].append(sample_name[0])
            samples["images"].append(image)
            true_segs.append(seg_mask)

            if not is_validation and save_images:
                utils.plot_sample(sample_name[0], image, seg_mask_predicted, seg_mask, save_folder, decision=prediction, plot_seg=plot_seg)
                utils.save_predicted_segmentation(seg_mask_predicted, sample_name[0], self.run_path)

        if is_validation:
            val_metrics = dict()
            metrics = utils.get_metrics(np.array(predictions_truths), np.array(predictions))
            FP, FN, TP, TN = list(map(sum, [metrics["FP"], metrics["FN"], metrics["TP"], metrics["TN"]]))
            self._log(f"VALIDATION on {eval_loader.dataset.kind} set || AUC={metrics['AUC']:f}, and AP={metrics['AP']:f}, with best thr={metrics['best_thr']:f} sat f-measure={metrics['best_f_measure']:.3f} and FP={FP:d}, FN={FN:d}, TOTAL SAMPLES={FP + FN + TP + TN:d}")

            decisions = np.array(predictions) >= metrics['best_thr']

            if self.cfg.SEG_BLACK:
                black_seg_counter = 0
                black_seg = np.zeros(predicted_segs[0].shape)
                for i, decision in enumerate(decisions):
                    if decision == False:
                        predicted_segs[i] = black_seg
                        black_seg_counter += 1
                self._log(f"Black Segmentations: {black_seg_counter}")

            # Dice
            step = 0.01
            dice = (0,0)
            iou = (0, 0)
            f1 = (0, 0)
            for i in range(len(predicted_segs)):
                true_segs[i] = np.array(true_segs[i]).astype(np.uint8)

            for thr in tqdm(np.arange(0.1, 1, step)):
                result_dice = []
                result_precision = []
                result_recall = []
                result_iou = []

                for i in range(len(predicted_segs)):
                    #y_true = np.array(true_segs[i]).astype(np.uint8)
                    y_true = true_segs[i]
                    y_pred = (np.array(predicted_segs[i])>thr).astype(np.uint8)

                    result_dice += [utils.dice(y_true, y_pred)]
                    result_precision += [utils.precision(y_true, y_pred)]
                    result_recall += [utils.recall(y_true, y_pred)]
                    result_iou += [utils.iou(y_true, y_pred)]

                if np.mean(result_dice) > dice[0]:
                    dice = (np.mean(result_dice), thr)
                
                if np.mean(result_iou) > iou[0]:
                    iou = (np.mean(result_iou), thr)
                
                f1_tmp = 2 * np.mean(result_precision) * np.mean(result_recall) / (np.mean(result_precision) + np.mean(result_recall))

                if f1_tmp > f1[0]:
                    f1 = (f1_tmp, thr)
                    val_metrics['Pr'] = np.mean(result_precision)
                    val_metrics['Re'] = np.mean(result_recall)

            self._log(f"Validation best Dice: {dice[0]:.4f} at {dice[1]:.3f}")
            self._log(f"Validation best IoU: {iou[0]:.4f} at {iou[1]:.3f}")
            self._log(f"Validation best F1: {f1[0]:.4f} at {f1[1]:.3f}")

            val_metrics['dec_threshold'] = metrics['best_thr']
            val_metrics['F1'], val_metrics['f1_threshold'] = f1
            val_metrics['Dice'], val_metrics['dice_threshold'] = dice
            val_metrics['IoU'], val_metrics['iou_threshold'] = iou
            val_metrics['best_f_measure'] = metrics['best_f_measure']

            return metrics["AP"], metrics["accuracy"], val_metrics
        else:
            decisions = np.array(predictions) >= thresholds["dec_threshold"]
            samples["decisions"] = list(decisions)
            FP, FN, TN, TP = utils.calc_confusion_mat(decisions, np.array(predictions_truths))

            fp = sum(FP).item()
            fn = sum(FN).item()
            tn = sum(TN).item()
            tp = sum(TP).item()

            pr = tp / (tp + fp) if tp else 0
            re = tp / (tp + fn) if tp else 0
            f1 = (2 * pr * re) / (pr + re) if pr and re else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            self._log(f"Decision EVAL on {eval_loader.dataset.kind}. Pr: {pr:f}, Re: {re:f}, F1: {f1:f}, Accuracy: {accuracy:f}, Threshold: {thresholds['dec_threshold']}")
            self._log(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

            # Max(S) classification
            seg_fp = 0
            seg_fn = 0
            seg_tp = 0
            seg_tn = 0
            for i in range(len(predictions_truths)):
                max_s = (predicted_segs[i] > thresholds["dice_threshold"]).astype(np.uint8).max()

                if max_s == 1:
                    if predictions_truths[i] == 1:
                        seg_tp += 1
                    elif predictions_truths[i] == 0:
                        seg_fp += 1
                elif max_s == 0:
                    if predictions_truths[i] == 1:
                        seg_fn += 1
                    elif predictions_truths[i] == 0:
                        seg_tn += 1
            self._log(f"Max(S) classification Pred crnenjem: TP: {seg_tp}, FP: {seg_fp}, FN: {seg_fn}, TN: {seg_tn}")

            if self.cfg.SEG_BLACK:
                black_seg_counter = 0
                black_seg = np.zeros(predicted_segs[0].shape)
                for i, decision in enumerate(decisions):
                    if decision == False:
                        if (predicted_segs[i] > thresholds["dice_threshold"]).astype(np.uint8).max() > 0:
                            black_seg_counter += 1
                            self._log(f"Blacked: {samples['image_names'][i]}\t{predictions_truths[i]}\t{true_segs[i].astype(np.uint8).max()}")
                        predicted_segs[i] = black_seg
                self._log(f"Black Segmentations: {black_seg_counter}")

            # Max(S) classification
            seg_fp = 0
            seg_fn = 0
            seg_tp = 0
            seg_tn = 0
            for i in range(len(predictions_truths)):
                max_s = (predicted_segs[i] > thresholds["dice_threshold"]).astype(np.uint8).max()

                if max_s == 1:
                    if predictions_truths[i] == 1:
                        seg_tp += 1
                    elif predictions_truths[i] == 0:
                        seg_fp += 1
                elif max_s == 0:
                    if predictions_truths[i] == 1:
                        seg_fn += 1
                    elif predictions_truths[i] == 0:
                        seg_tn += 1
            self._log(f"Max(S) classification Po crnenjem: TP: {seg_tp}, FP: {seg_fp}, FN: {seg_fn}, TN: {seg_tn}")
            
            # Dice, IoU in F1
            mean_dice, std_dice, mean_iou, std_iou, mean_pr, std_pr, mean_re, std_re, adj_thr_c = utils.dice_iou(predicted_segs, true_segs, thresholds, samples["images"], samples["image_names"], self.run_path, decisions, save_images=self.cfg.SAVE_IMAGES, adjusted_threshold=self.cfg.THR_ADJUSTMENT)
            
            # Adjusted threshold
            if self.cfg.THR_ADJUSTMENT:
                self._log(f"Adjusted thresholds: {adj_thr_c}")

            self._log(f"{eval_loader.dataset.kind} set. Precision mean = {mean_pr:f}, std = {std_pr:f}")
            self._log(f"{eval_loader.dataset.kind} set. Recall mean = {mean_re:f}, std = {std_re:f}")
            self._log(f"{eval_loader.dataset.kind} set. F1 mean = {2 * mean_pr * mean_re / (mean_pr + mean_re):f}, std = {2 * std_pr * std_re / (std_pr + std_re):f} at {thresholds['f1_threshold']:f}")
            self._log(f"{eval_loader.dataset.kind} set. Dice mean = {mean_dice:f}, std = {std_dice:f} at {thresholds['dice_threshold']:f}")
            self._log(f"{eval_loader.dataset.kind} set. IoU mean = {mean_iou:f}, std = {std_iou:f} at {thresholds['iou_threshold']:f}")

    def get_dec_gradient_multiplier(self):
        if self.cfg.GRADIENT_ADJUSTMENT:
            grad_m = 0
        else:
            grad_m = 1

        self._log(f"Returning dec_gradient_multiplier {grad_m}", LVL_DEBUG)
        return grad_m

    def set_dec_gradient_multiplier(self, model, multiplier):
        model.set_gradient_multipliers(multiplier)

    def get_loss_weights(self, epoch):
        total_epochs = float(self.cfg.EPOCHS)

        if self.cfg.DYN_BALANCED_LOSS:
            seg_loss_weight = 1 - (epoch / total_epochs)
            dec_loss_weight = self.cfg.DELTA_CLS_LOSS * (epoch / total_epochs)
        else:
            seg_loss_weight = 1
            dec_loss_weight = self.cfg.DELTA_CLS_LOSS

        self._log(f"Returning seg_loss_weight {seg_loss_weight} and dec_loss_weight {dec_loss_weight}", LVL_DEBUG)
        return seg_loss_weight, dec_loss_weight

    def reload_model(self, model, load_final=False):
        if self.cfg.USE_BEST_MODEL:
            if self.cfg.BEST_MODEL_TYPE == "dec":
                path = os.path.join(self.model_path, "best_dec_dict.pth")
            elif self.cfg.BEST_MODEL_TYPE == "seg":
                path = os.path.join(self.model_path, "best_seg_dict.pth")
            else:
                path = os.path.join(self.model_path, "best_state_dict.pth")
            model.load_state_dict(torch.load(path, map_location=f"cuda:{self.cfg.GPU}"))
            self._log(f"Loading model state from {path}")
        elif load_final:
            path = os.path.join(self.model_path, "final_state_dict.pth")
            model.load_state_dict(torch.load(path, map_location=f"cuda:{self.cfg.GPU}"))
            self._log(f"Loading model state from {path}")
        else:
            self._log("Keeping same model state")

    def _save_params(self):
        params = self.cfg.get_as_dict()
        params_lines = sorted(map(lambda e: e[0] + ":" + str(e[1]) + "\n", params.items()))
        fname = os.path.join(self.run_path, "run_params.txt")
        with open(fname, "w+") as f:
            f.writelines(params_lines)

    def _save_train_results(self, results):
        losses, validation_data, validation_metrics, lrs = results
        ls, ld, l, le = map(list, zip(*losses))
        plt.plot(le, l, label="Loss", color="red")
        plt.plot(le, ls, label="Loss seg")
        plt.plot(le, ld, label="Loss dec")
        plt.ylim(bottom=0)
        plt.grid()
        plt.xlabel("Epochs")
        if self.cfg.VALIDATE:
            v, ve = map(list, zip(*validation_data))
            plt.twinx()
            plt.plot(ve, v, label="Validation AP", color="Green")
            plt.ylim((0, 1))
        plt.legend()
        plt.savefig(os.path.join(self.run_path, "loss_val"), dpi=200)

        df_loss = pd.DataFrame(data={"loss_seg": ls, "loss_dec": ld, "loss": l, "epoch": le})
        df_loss.to_csv(os.path.join(self.run_path, "losses.csv"), index=False)

        if self.cfg.VALIDATE:
            df_loss = pd.DataFrame(data={"validation_data": ls, "loss_dec": ld, "loss": l, "epoch": le})
            df_loss.to_csv(os.path.join(self.run_path, "losses.csv"), index=False)
        
        # Dice & IOU plot
        if len(validation_metrics) != 0:
            epochs, metrics = map(list, zip(*validation_metrics))
            f1 = [i['F1'] for i in metrics]
            pr = [i['Pr'] for i in metrics]
            re = [i['Re'] for i in metrics]
            dice = [i['Dice'] for i in metrics]
            iou = [i['IoU'] for i in metrics]
            plt.clf()
            plt.plot(epochs, f1, label="F1")
            plt.plot(epochs, pr, label="Pr")
            plt.plot(epochs, re, label="Re")
            plt.plot(epochs, dice, label="Dice")
            plt.plot(epochs, iou, label="IoU")
            plt.xlabel("Epochs")
            plt.ylabel("Score")
            plt.legend()
            plt.savefig(os.path.join(self.run_path, "scores"), dpi=200)

        # Loss plot
        # Loss
        plt.clf()
        plt.plot(le, l)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.run_path, "loss"), dpi=200)
        
        # Loss Segmentation
        plt.clf()
        plt.plot(le, ls)
        plt.xlabel("Epochs")
        plt.ylabel("Loss Segmentation")
        plt.savefig(os.path.join(self.run_path, "loss_seg"), dpi=200)

        # Loss Dec
        plt.clf()
        plt.plot(le, ld)
        plt.xlabel("Epochs")
        plt.ylabel("Loss Dec")
        plt.savefig(os.path.join(self.run_path, "loss_dec"), dpi=200)

        # Learning rate plot
        epochs, lr = map(list, zip(*lrs))
        plt.clf()
        plt.plot(epochs, lr)
        plt.xlabel("Epochs")
        plt.ylabel("Learning rate")
        plt.savefig(os.path.join(self.run_path, "learning_rate"), dpi=200)

    def _save_model(self, model, name="final_state_dict.pth"):
        output_name = os.path.join(self.model_path, name)
        self._log(f"Saving current model state to {output_name}")
        if os.path.exists(output_name):
            os.remove(output_name)

        torch.save(model.state_dict(), output_name)

    def _get_optimizer(self, model):
        if self.cfg.OPTIMIZER == "sgd":
            return torch.optim.SGD(model.parameters(), self.cfg.LEARNING_RATE)
        elif self.cfg.OPTIMIZER == "adam":
            return torch.optim.Adam(model.parameters(), self.cfg.LEARNING_RATE)

    def _get_scheduler(self, optimizer):
        if self.cfg.SCHEDULER is None:
            return None
        else:
            self._log(f"Using Learning Rate Scheduler: StepLR, Step size: {int(self.cfg.SCHEDULER[0])}, Gamma: {self.cfg.SCHEDULER[1]}")
            return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=int(self.cfg.SCHEDULER[0]), gamma=self.cfg.SCHEDULER[1])

    def _get_learning_rate(self, optimizer):
        for p in optimizer.param_groups:
            return p["lr"]

    def _get_loss(self, is_seg, pos_weight=None):
        reduction = "none" if self.cfg.WEIGHTED_SEG_LOSS and is_seg else "mean"
        if self.cfg.BCE_LOSS_W and pos_weight is not None:
            return nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=torch.Tensor([pos_weight])).to(self._get_device())
        else:
            return nn.BCEWithLogitsLoss(reduction=reduction).to(self._get_device())

    def _get_device(self):
        return f"cuda:{self.cfg.GPU}"

    def _set_results_path(self):
        self.run_name = f"{self.cfg.RUN_NAME}_FOLD_{self.cfg.FOLD}" if self.cfg.DATASET in ["KSDD", "DAGM"] else self.cfg.RUN_NAME

        results_path = os.path.join(self.cfg.RESULTS_PATH, self.cfg.DATASET)
        self.tensorboard_path = os.path.join(results_path, "tensorboard", self.run_name)

        run_path = os.path.join(results_path, self.cfg.RUN_NAME)
        if self.cfg.DATASET in ["KSDD", "DAGM"]:
            run_path = os.path.join(run_path, f"FOLD_{self.cfg.FOLD}")

        self._log(f"Executing run with path {run_path}")

        self.run_path = run_path
        self.model_path = os.path.join(run_path, "models")
        self.outputs_path = os.path.join(run_path, "test_outputs")

    def _create_results_dirs(self):
        list(map(utils.create_folder, [self.run_path, self.model_path, self.outputs_path, ]))

    def _get_model(self):
        if self.cfg.ARCHITECTURE == 'SegDecNetOriginalJIM':
            seg_net = SegDecNetOriginalJIM(self._get_device(), self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_CHANNELS)
        elif self.cfg.ARCHITECTURE == 'SegDecNet++':
            seg_net = SegDecNetPlusPlus(self._get_device(), self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_CHANNELS)
        else:
            raise Exception('Invalid architecture requested')
        return seg_net

    def print_run_params(self):
        for l in sorted(map(lambda e: e[0] + ":" + str(e[1]) + "\n", self.cfg.get_as_dict().items())):
            k, v = l.split(":")
            self._log(f"{k:25s} : {str(v.strip())}")
    
    def set_seed(self):
        if self.cfg.REPRODUCIBLE_RUN is not None:
            self._log(f"Reproducible run, fixing all seeds to: {self.cfg.REPRODUCIBLE_RUN}", LVL_DEBUG)
            np.random.seed(self.cfg.REPRODUCIBLE_RUN)
            torch.manual_seed(self.cfg.REPRODUCIBLE_RUN)
            random.seed(self.cfg.REPRODUCIBLE_RUN)
            torch.cuda.manual_seed(self.cfg.REPRODUCIBLE_RUN)
            torch.cuda.manual_seed_all(self.cfg.REPRODUCIBLE_RUN)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
    # def seg_val_metrics(self, truth_segmentations, predicted_segmentations, dataset_kind, threshold_step=0.005, pxl_distance=2):
    #     n_samples = len(truth_segmentations)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1 + pxl_distance * 2, 1 + pxl_distance * 2))
    #     thresholds, pr_results, re_results, f1_results = [], [], [], []
    #     metrics = dict()
    #
    #     self._log(f"Validation metrics on {dataset_kind} set. {pxl_distance} pixel distance used. Threshold step: {threshold_step}")
    #
    #     for threshold in np.arange(0.1, 1, threshold_step):
    #         results = []
    #         for i in range(n_samples):
    #             y_true = np.array(truth_segmentations[i]).astype(np.uint8)
    #             y_true_d = cv2.dilate(y_true, kernel) if pxl_distance > 0 else y_true
    #             y_pred = (np.array(predicted_segmentations[i])>threshold).astype(np.uint8)
    #
    #             tp_d = sum(sum((y_true_d==1)&(y_pred==1))).item()
    #             fp_d = sum(sum((y_true_d==0)&(y_pred==1))).item()
    #             fn = sum(sum((y_true==1)&(y_pred==0))).item()
    #
    #             pr = tp_d / (tp_d + fp_d) if tp_d else 0
    #             re = tp_d / (tp_d + fn) if tp_d else 0
    #             f1 = (2 * pr * re) / (pr + re) if pr and re else 0
    #
    #             results.append((pr, re, f1))
    #
    #         thresholds.append(threshold)
    #         pr_results.append(np.mean(np.array(results)[:, 0]))
    #         re_results.append(np.mean(np.array(results)[:, 1]))
    #         f1_results.append(np.mean(np.array(results)[:, 2]))
    #
    #     f1_max_index = f1_results.index(max(f1_results))
    #     metrics['Pr'] = pr_results[f1_max_index]
    #     metrics['Re'] = re_results[f1_max_index]
    #     metrics['F1'] = max(f1_results)
    #     metrics['f1_threshold'] = thresholds[f1_max_index]
    #
    #     self._log(f"Best F1: {metrics['F1']:f} at {thresholds[f1_max_index]:f}. Pr: {metrics['Pr']:f}, Re: {metrics['Re']:f}")
    #
    #     return metrics