import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
import datetime

class DetectFailureCase:
    def __init__(self, opt, motion_autoencoder, data, res_video_path):
        self.opt = opt

        self.res_video_path = res_video_path
        call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.failure_case_dir = os.path.join(opt.res_dir, f"failure_case_{call_time}")
        os.makedirs(self.failure_case_dir, exist_ok=True)

        self.motion_autoencoder = motion_autoencoder
        self.data = data

        self.threshold_frob_err = 0.05
        self.threshold_explained_var = 0.9
        self.threshold_avg_cos = 0.9
        self.threshold_avg_std_diff = 1e-4
        self.threshold_avg_std_ori = 0.2

    def numpy_to_python(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj  # let json handle other native types

    def coeff_to_latent(self, Q, coeff):
        """
        Convert coeff to latent
        Q: (512, 20)
        coeff: (num_sample, 20)
        output: (num_sample, 512)
        """
        input_diag = torch.diag_embed(coeff)  # alpha, diagonal matrix
        out = torch.matmul(input_diag, Q.T)
        out = torch.sum(out, dim=1)
        return out

    def plot_stats_steps(self, dict_metrics, fps=25):
        plot_dict = {'current': dict_metrics}
        color_dict = {'current': 'red'}
        metric_list = ['std', 'length']

        # Only 1 row per metric now (diff only)
        fig, axes = plt.subplots(len(metric_list), 1, figsize=(12, 2 * 2), sharex=True)

        if len(metric_list) == 1:
            axes = [axes]  # make iterable if single metric

        for i, metric in enumerate(metric_list):
            for key, dict_metric in plot_dict.items():
                num_frames = len(dict_metric['cos'])
                diff = np.array(dict_metric["sample_ori"][metric]) - np.array(dict_metric["sample_recon"][metric])

                # Detect step changes
                changepoints = self.sliding_window_change(diff, window=10, thr_mul=2, min_length=2)

                # Plot diff
                axes[i].plot(range(num_frames), diff, color=color_dict[key], label=f"{key} diff")
                axes[i].set_ylabel(f"{metric} (diff)")
                axes[i].grid(True, linestyle="--", alpha=0.5)

                # Mark detected changepoints
                for cp in changepoints:
                    axes[i].axvline(cp, color='magenta', linestyle='--', alpha=0.8)
                    axes[i].text(cp, axes[i].get_ylim()[1] * 0.9,
                                f"{cp//fps}s", rotation=90,
                                va='top', ha='center', color='magenta', fontsize=8)

            # Common x-axis ticks (time in seconds)
            frame_ticks = np.arange(0, num_frames, fps)
            time_labels = (frame_ticks / fps).astype(int)
            axes[i].set_xlim(0, num_frames - 1)
            axes[i].set_xticks(frame_ticks)
            axes[i].set_xticklabels(time_labels)
            axes[i].set_xlabel("Time (s)")
            axes[i].legend()

        fig.suptitle("std/length difference with step detection", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        output_path = os.path.join(self.failure_case_dir, "std_length_diff_steps.png")
        plt.savefig(output_path)
        print(f"Plot saved at {output_path}")

    @torch.no_grad()
    def organize_saved_result(self, X, X_hat, dict_detect_failure_metrics, changepoints_std, changepoints_length, broken_flag, ghost_flag, weird_motion_flag, error_msg):
        # pickle for detailed result
        output_dict = {"per_frame": {}, "average_over_frames": {}}
        changepoints_dict = {
            "changepoints_std": changepoints_std,
            "changepoints_length": changepoints_length,
            "change_frequency_std": len(changepoints_std) / X.shape[1],
            "change_frequency_length": len(changepoints_length) / X.shape[1],
        }
        failure_flag_dict = {
            "broken_flag": broken_flag,
            "ghost_flag": ghost_flag,
            "weird_motion_flag": weird_motion_flag,
            "error_msg": error_msg,
        }
        output_dict["changepoints"] = changepoints_dict
        output_dict["failure_flag"] = failure_flag_dict
        output_dict["sample_ori"] = X.cpu().numpy()
        output_dict["sample_recon"] = X_hat.cpu().numpy()
        output_dict["per_frame"]["sample_ori"] = dict_detect_failure_metrics["sample_ori"]
        output_dict["per_frame"]["sample_recon"] = dict_detect_failure_metrics["sample_recon"]
        output_dict["per_frame"]["frob_err"] = dict_detect_failure_metrics["frob_err"]
        output_dict["per_frame"]["cos"] = dict_detect_failure_metrics["cos"]
        output_dict["average_over_frames"]["avg_frob_err"] = dict_detect_failure_metrics["avg_frob_err"]
        output_dict["average_over_frames"]["avg_explained_var"] = dict_detect_failure_metrics["avg_explained_var"]
        output_dict["average_over_frames"]["avg_cos"] = dict_detect_failure_metrics["avg_cos"]
        output_dict["average_over_frames"]["avg_std_all_frames_ori"] = dict_detect_failure_metrics["avg_std_all_frames_ori"]
        output_dict["average_over_frames"]["avg_std_all_frames_recon"] = dict_detect_failure_metrics["avg_std_all_frames_recon"]
        output_dict["average_over_frames"]["avg_length_all_frames_ori"] = dict_detect_failure_metrics["avg_length_all_frames_ori"]
        output_dict["average_over_frames"]["avg_length_all_frames_recon"] = dict_detect_failure_metrics["avg_length_all_frames_recon"]

        # json for final result
        output_dict_json = {
            "average_over_frames": output_dict["average_over_frames"],
            "failure_flag": output_dict["failure_flag"],
            "changepoints": output_dict["changepoints"],
        }
        return output_dict, output_dict_json

    def sliding_window_change(self, vals, window=5, thr_mul=2.5, min_length=3):
        """
        Detect steps by comparing means of adjacent windows.
        Returns indices of detected change-points.
        """
        N = len(vals)
        diff_means = np.zeros(N)
        for t in range(window, N - window):
            left = vals[t - window:t].mean()
            right = vals[t:t + window].mean()
            diff_means[t] = right - left

        sigma = np.std(diff_means)
        mask = np.abs(diff_means) > (thr_mul * sigma)

        inds = np.where(mask)[0]
        if inds.size == 0:
            return []
        groups = np.split(inds, np.where(np.diff(inds) != 1)[0] + 1)
        changepoints = [g[0] for g in groups if len(g) >= min_length]
        return changepoints

    @torch.no_grad()
    def detect_failure(self, X: torch.Tensor):
        """
        X: (512, T)
        """
        X = X.clone().squeeze(0).t()  # X == r_s (512, T)

        # reconstruct X
        Q, _ = torch.linalg.qr(self.motion_autoencoder.dec.direction.weight) # (512, 20)
        coeff = torch.matmul(Q.t(), X) # (20, 512)(512, T): (512, T) -> (20, T)
        X_hat = self.coeff_to_latent(Q, coeff.t()).t() # (20, T) -> (512, T)

        dict_detect_failure_metrics = {
            "sample_ori": {"mean": [], "std": [], "length": []},
            "sample_recon": {"mean": [], "std": [], "length": []},
            "frob_err": [],
            "cos": [],
        }

        # broken (average)
        # Reconstruction error
        frob_err = torch.norm(X - X_hat, p='fro') / torch.norm(X, p='fro').item()

        # Explained variance
        var_total = torch.norm(X, p='fro')**2
        var_recon = torch.norm(X_hat, p='fro')**2
        explained_var = (var_recon / var_total).item()

        # Cosine similarity per frame
        cos_sims = []
        for t in range(X.shape[1]):
            cos = torch.nn.functional.cosine_similarity(X[:, t], X_hat[:, t], dim=0)
            cos_sims.append(cos.item())
        avg_cos = sum(cos_sims) / len(cos_sims)

        # Average 512-dim std of all frames
        ori_avg_std = torch.mean(torch.std(X, axis=1)).item()
        recon_avg_std = torch.mean(torch.std(X_hat, axis=1)).item()

        dict_detect_failure_metrics['avg_frob_err'] = float(frob_err)
        dict_detect_failure_metrics['avg_explained_var'] = float(explained_var)
        dict_detect_failure_metrics['avg_cos'] = float(avg_cos)
        dict_detect_failure_metrics["avg_std_all_frames_ori"] = float(ori_avg_std)
        dict_detect_failure_metrics["avg_std_all_frames_recon"] = float(recon_avg_std)


        # ghost (frame-by-frame)
        for t in range(X.shape[1]):
            # original sample
            X_t = X[:, t]
            X_mean_t = torch.mean(X_t)
            X_std_t = torch.std(X_t)
            X_length_t = torch.norm(X_t)

            # reconstructed sample
            X_hat_t = X_hat[:, t]
            X_hat_mean_t = torch.mean(X_hat_t)  # (512,)
            X_hat_std_t = torch.std(X_hat_t)  # (512,)
            X_hat_length_t = torch.norm(X_hat_t)  # (512,)

            # Reconstruction error
            frob_err = torch.norm(X_t - X_hat_t, p='fro') / torch.norm(X_t, p='fro')

            # Cosine similarity per frame
            cos = torch.nn.functional.cosine_similarity(X_t, X_hat_t, dim=0)

            dict_detect_failure_metrics["sample_ori"]["mean"].append(X_mean_t.item())
            dict_detect_failure_metrics["sample_ori"]["std"].append(X_std_t.item())
            dict_detect_failure_metrics["sample_ori"]["length"].append(X_length_t.item())
            dict_detect_failure_metrics["sample_recon"]["mean"].append(X_hat_mean_t.item())
            dict_detect_failure_metrics["sample_recon"]["std"].append(X_hat_std_t.item())
            dict_detect_failure_metrics["sample_recon"]["length"].append(X_hat_length_t.item())
            dict_detect_failure_metrics["frob_err"].append(frob_err.item())
            dict_detect_failure_metrics["cos"].append(cos.item())


        ori_avg_length = np.mean(dict_detect_failure_metrics["sample_ori"]["length"])
        recon_avg_length = np.mean(dict_detect_failure_metrics["sample_recon"]["length"])
        # TODO: check threshould for ori_avg_length and recon_avg_length
        # print(f"ori_avg_length: {ori_avg_length}, recon_avg_length: {recon_avg_length}")
        dict_detect_failure_metrics["avg_length_all_frames_ori"] = float(ori_avg_length)
        dict_detect_failure_metrics["avg_length_all_frames_recon"] = float(recon_avg_length)

        # Use std_diff and length_diff to detect the exact ghost timepoints
        std_diff = torch.tensor(dict_detect_failure_metrics["sample_ori"]["std"]) - torch.tensor(dict_detect_failure_metrics["sample_recon"]["std"])
        length_diff = torch.tensor(dict_detect_failure_metrics["sample_ori"]["length"]) - torch.tensor(dict_detect_failure_metrics["sample_recon"]["length"])

        # Detect step changes
        changepoints_std = self.sliding_window_change(std_diff, window=10, thr_mul=2, min_length=2)
        changepoints_length = self.sliding_window_change(length_diff, window=10, thr_mul=2, min_length=2)

        # check metrics
        broken_flag = False
        ghost_flag = False
        weird_motion_flag = False
        error_msg = ""

        if frob_err > self.threshold_frob_err:
            error_msg += f"frob_err is too large (> {self.threshold_frob_err:.4f}): {frob_err:.4f} \n"
            broken_flag = True
        if explained_var < self.threshold_explained_var:
            error_msg += f"explained_var is too small (< {self.threshold_explained_var:.4f}): {explained_var:.4f} \n"
            broken_flag = True
        if avg_cos < self.threshold_avg_cos:
            error_msg += f"avg_cos is too small (< {self.threshold_avg_cos:.4f}): {avg_cos:.4f} \n"
            broken_flag = True

        if abs(ori_avg_std - recon_avg_std) < self.threshold_avg_std_diff:
            # TODO: check threshould for ori_avg_length and recon_avg_length
            if ori_avg_std > self.threshold_avg_std_ori:
                error_msg += f"avg_std_all_frames is too large (> {self.threshold_avg_std_ori:.4f}): {ori_avg_std:.4f} \n"
                if len(changepoints_std) > 0 and len(changepoints_length) > 0:
                    ghost_flag = True
        else:
            error_msg += f"avg_std_all_frames is different between orig ({ori_avg_std:.4f}) and recon ({recon_avg_std:.4f}) -> weird motion \n"
            weird_motion_flag = True

        dict_detect_failure_metrics["error_msg"] = str(error_msg)

        # plot detected ghost
        output_dict, output_dict_json = self.organize_saved_result(X, X_hat, dict_detect_failure_metrics, changepoints_std, changepoints_length, broken_flag, ghost_flag, weird_motion_flag, error_msg)
        self.plot_stats_steps(dict_detect_failure_metrics)

        # save result
        self.save_result(output_dict, output_dict_json)
        print(f"Detect result saved at {self.failure_case_dir}")

    def save_result(self, output_dict, output_dict_json):
        output_dict['image'] = self.opt.ref_path
        output_dict['audio'] = self.opt.aud_path
        output_dict['audio_wav2vec_feature'] = self.data['a'].detach().cpu().numpy()
        output_dict['model'] = self.opt.ckpt_path
        output_dict['video_result'] = self.res_video_path
        output_dict['no_crop'] = self.opt.no_crop

        use_fake_silence = False
        if np.all(self.data['a'].detach().cpu().numpy() == 0):
            use_fake_silence = True

        output_dict_json.update({
            'image': self.opt.ref_path,
            'audio': self.opt.aud_path,
            'model': self.opt.ckpt_path,
            'video_result': self.res_video_path,
            'use_fake_silence': use_fake_silence,
            'no_crop': self.opt.no_crop,
            }
        )
        print(json.dumps(output_dict_json, indent=2, default=self.numpy_to_python))

        detect_result_pickle_path = os.path.join(self.failure_case_dir, "detect_result.pkl")
        with open(detect_result_pickle_path, "wb") as f:
            pickle.dump(output_dict, f)

        detect_result_json_path = os.path.join(self.failure_case_dir, "detect_result.json")
        with open(detect_result_json_path, "w") as f:
            json.dump(output_dict_json, f, default=self.numpy_to_python, indent=2)

        import shutil
        shutil.copy(self.res_video_path, os.path.join(self.failure_case_dir, "video_result.mp4"))