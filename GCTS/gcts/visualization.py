import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np



def plot_time_series_reconstruction(
    originals,
    reconstructions,
    ts_indices,
    save_dir="ts_plots",
    tag="",
    anomaly_labels=None,
):
    """
    Plot original vs reconstructed time series for selected node indices
    and save each as a .png file.
 
    Args:
        originals       : np.ndarray of shape (T_total, num_nodes)
                          Full concatenated original signal across all windows.
        reconstructions : np.ndarray of shape (T_total, num_nodes)
                          Full concatenated reconstructed signal across all windows.
        ts_indices      : list[int]  — which node/TS indices to plot (e.g. [0,1,2]).
        save_dir        : str        — directory where .png files are written.
        tag             : str        — optional string appended to filenames (e.g. "epoch5").
        anomaly_labels  : np.ndarray of shape (T_total,) or None.
                          If provided, shades anomalous timesteps in red.
    """
    os.makedirs(save_dir, exist_ok=True)
    T = originals.shape[0]
    time_axis = np.arange(T)
 
    for idx in ts_indices:
        if idx >= originals.shape[1]:
            print(f"[plot_time_series_reconstruction] Warning: ts_index {idx} "
                  f"out of range (num_nodes={originals.shape[1]}), skipping.")
            continue
 
        fig, ax = plt.subplots(figsize=(14, 4))
 
        if anomaly_labels is not None:
            in_anomaly = False
            start = 0
            for t in range(T):
                if anomaly_labels[t] == 1 and not in_anomaly:
                    start = t
                    in_anomaly = True
                elif anomaly_labels[t] == 0 and in_anomaly:
                    ax.axvspan(start, t, color="red", alpha=0.15, label="_nolegend_")
                    in_anomaly = False
            if in_anomaly:
                ax.axvspan(start, T, color="red", alpha=0.15, label="Anomaly")
 
        ax.plot(time_axis, originals[:, idx],
                label="Original", color="steelblue", linewidth=1.2)
        ax.plot(time_axis, reconstructions[:, idx],
                label="Reconstructed", color="darkorange",
                linewidth=1.0, linestyle="--", alpha=0.85)
 
        ax.set_title(f"Time Series node {idx}  —  {tag}", fontsize=13)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
 
        fname_tag = f"_{tag}" if tag else ""
        save_path = os.path.join(save_dir, f"ts_node{idx}{fname_tag}.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"[plot_time_series_reconstruction] Saved → {save_path}")
 

def plot_reconstruction_error_over_time(
    struct_error_per_window,      # was: error_per_window
    temp_error_per_window,        # new
    labels_per_point,
    window_size,
    save_dir="ts_plots",
    tag="",
):
    os.makedirs(save_dir, exist_ok=True)

    struct_errors = np.asarray(struct_error_per_window)
    temp_errors   = np.asarray(temp_error_per_window)
    labels        = np.asarray(labels_per_point).astype(int)

    # Expand window-level scores to point-level
    struct_expanded = np.repeat(struct_errors, window_size)
    temp_expanded   = np.repeat(temp_errors,   window_size)

    min_len = min(len(struct_expanded), len(temp_expanded), len(labels))
    struct_expanded = struct_expanded[:min_len]
    temp_expanded   = temp_expanded[:min_len]
    labels          = labels[:min_len]
    time_axis       = np.arange(min_len)

    fig, (ax_struct, ax_temp) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax_struct.plot(time_axis, struct_expanded, linewidth=1.2,
                   color="steelblue", label="Structural error")
    thresh_struct = struct_expanded.mean() + struct_expanded.std()
    ax_struct.axhline(thresh_struct, linestyle=":", linewidth=1, color="steelblue",
                      label=f"Threshold (μ+σ={thresh_struct:.4f})")
    ax_struct.set_ylabel("Error", color="steelblue")
    ax_struct.tick_params(axis="y", labelcolor="steelblue")
    ax_struct.set_title(f"Spatial reconstruction error — {tag}", fontsize=12)
    ax_struct.grid(alpha=0.3)

    ax_s2 = ax_struct.twinx()
    ax_s2.fill_between(time_axis, labels, alpha=0.20, color="red", label="Anomaly")
    ax_s2.set_ylim(-0.1, 1.1)
    ax_s2.set_ylabel("Label", color="red")
    ax_s2.tick_params(axis="y", labelcolor="red")

    lines1, labs1 = ax_struct.get_legend_handles_labels()
    lines2, labs2 = ax_s2.get_legend_handles_labels()
    ax_struct.legend(lines1 + lines2, labs1 + labs2, loc="upper left")

    ax_temp.plot(time_axis, temp_expanded, linewidth=1.2,
                 color="darkorange", label="Temporal error")
    thresh_temp = temp_expanded.mean() + temp_expanded.std()
    ax_temp.axhline(thresh_temp, linestyle=":", linewidth=1, color="darkorange",
                    label=f"Threshold (μ+σ={thresh_temp:.4f})")
    ax_temp.set_ylabel("Error", color="darkorange")
    ax_temp.tick_params(axis="y", labelcolor="darkorange")
    ax_temp.set_title(f"Temporal reconstruction error — {tag}", fontsize=12)
    ax_temp.set_xlabel("Time index")
    ax_temp.grid(alpha=0.3)

    ax_t2 = ax_temp.twinx()
    ax_t2.fill_between(time_axis, labels, alpha=0.20, color="red", label="Anomaly")
    ax_t2.set_ylim(-0.1, 1.1)
    ax_t2.set_ylabel("Label", color="red")
    ax_t2.tick_params(axis="y", labelcolor="red")

    lines3, labs3 = ax_temp.get_legend_handles_labels()
    lines4, labs4 = ax_t2.get_legend_handles_labels()
    ax_temp.legend(lines3 + lines4, labs3 + labs4, loc="upper left")

    fname_tag = f"_{tag}" if tag else ""
    save_path = os.path.join(save_dir, f"error_over_time{fname_tag}.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[plot_reconstruction_error_over_time] Saved → {save_path}")

def plot_the_curves(val_scores, test_scores, save_path):
    print("Plotting training curves")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(val_scores, label="Validation Score")
    ax.plot(test_scores, label="Test Score")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True)
    
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return
