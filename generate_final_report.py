"""Generate the Final Report covering all six stages of the AI & NN course project."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF

root = Path(__file__).parent
out_path = root / "Final_Report.pdf"

# ---- Figure paths ---------------------------------------------------------
ass1_fig = root / "Assigment1" / "figures"
ass2_fig = root / "Assigment2" / "figures"
ass4_fig = root / "Assignment4" / "figures"
ass5_fig = root / "Assignment5" / "images"
ass6_fig = root / "Assignment6" / "figures"

# ---- Cross-stage metrics --------------------------------------------------
stage23_df = pd.DataFrame(
    [
        {"Model": "MLP (Stage 2)", "Test RMSE": 71.2991, "Test MAE": 58.9493},
        {"Model": "DNN (Stage 2)", "Test RMSE": 67.6085, "Test MAE": 55.1771},
        {"Model": "LSTM window=12 (Stage 3, saved)", "Test RMSE": 59.7064, "Test MAE": 49.9229},
        {"Model": "LSTM window=48 (Stage 3, best)", "Test RMSE": 54.0375, "Test MAE": 44.0371},
    ]
)
energy_df = pd.read_csv(root / "Assignment6" / "model_comparison.csv")

# Regression improvement plot for stages 2 / 3
fig, ax = plt.subplots(figsize=(8, 4))
stage23_df.set_index("Model")[["Test RMSE", "Test MAE"]].plot(kind="bar", ax=ax, rot=20)
ax.set_title("SCANIA Regression (Stages 2 & 3) - Test RMSE / MAE")
ax.set_ylabel("Error (lower is better)")
plt.tight_layout()
plt.savefig(root / "figures_final_regression.png", dpi=140)
plt.close()

# Energy comparison plot
fig, ax = plt.subplots(figsize=(9, 4.5))
energy_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot(
    kind="bar", ax=ax, rot=25, ylim=(0, 1)
)
ax.set_title("Energy Dataset (Stages 4 / 5 / 6) - Test Metrics")
ax.set_ylabel("Score")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(root / "figures_final_energy.png", dpi=140)
plt.close()


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, "AI & Neural Networks - Course Project Final Report",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, t):
        self.set_font("Helvetica", "B", 15); self.ln(4)
        self.cell(0, 10, t, new_x="LMARGIN", new_y="NEXT"); self.ln(2)

    def sub_title(self, t):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, t, new_x="LMARGIN", new_y="NEXT"); self.ln(1)

    def sub_sub_title(self, t):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 7, t, new_x="LMARGIN", new_y="NEXT"); self.ln(1)

    def body_text(self, t):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, t); self.ln(2)

    def bullet(self, t):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, "  - " + t); self.ln(0)

    def add_figure(self, path, caption, w=170):
        if not path.exists():
            self.body_text(f"[figure missing: {path.name}]")
            return
        self.ln(2)
        x = (210 - w) / 2
        self.image(str(path), x=x, w=w)
        self.ln(2)
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def simple_table(self, headers, rows, col_w):
        self.set_font("Helvetica", "B", 9)
        for i, h in enumerate(headers):
            self.cell(col_w[i], 7, h, border=1, align="C")
        self.ln()
        self.set_font("Helvetica", "", 9)
        for row in rows:
            for i, v in enumerate(row):
                self.cell(col_w[i], 7, str(v), border=1, align="C")
            self.ln()
        self.ln(3)


pdf = Report()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ==========================================================================
# Title page
# ==========================================================================
pdf.add_page()
pdf.ln(45)
pdf.set_font("Helvetica", "B", 22)
pdf.cell(0, 12, "AI & Neural Networks", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 12, "Course Project - Final Report", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)
pdf.set_font("Helvetica", "", 15)
pdf.cell(0, 9, "From Predictive Maintenance to Energy Monitoring:", align="C",
         new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 9, "Six Stages of Deep Learning on Time-Series Data", align="C",
         new_x="LMARGIN", new_y="NEXT")
pdf.ln(25)
pdf.set_font("Helvetica", "", 11)
pdf.cell(0, 7, "Course: AI & Neural Networks", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "University: Astana IT University (AITU)", align="C",
         new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Date: April 2026", align="C", new_x="LMARGIN", new_y="NEXT")

# ==========================================================================
# Abstract
# ==========================================================================
pdf.add_page()
pdf.section_title("Abstract")
pdf.body_text(
    "This report summarises a six-stage course project in which the same team of students went "
    "end-to-end from raw time-series to a deployable anomaly-detection module. Stages 1 to 3 "
    "were developed on the SCANIA Component X predictive-maintenance dataset, where the goal "
    "was regression on a time-to-event target. At Stage 4 the team switched to the Appliance "
    "Energy Prediction dataset, converted the problem to binary classification above/below the "
    "median consumption, and retained the same chronological 70/15/15 split for the remainder "
    "of the project. Across the six stages we implemented and compared MLP, DNN, LSTM, 1D CNN, "
    "a denoising autoencoder + CNN pipeline, and a Siamese network trained with a hand-coded "
    "contrastive loss. The report collects the key metrics, the architectures, the qualitative "
    "observations, and the integrated analysis across all stages."
)

# ==========================================================================
# Table of contents (manual)
# ==========================================================================
pdf.section_title("Contents")
pdf.body_text(
    "1. Project overview and datasets\n"
    "2. Stage 1 - EDA and feature engineering (SCANIA)\n"
    "3. Stage 2 - MLP and DNN regression (SCANIA)\n"
    "4. Stage 3 - LSTM regression (SCANIA)\n"
    "5. Stage 4 - 1D CNN binary classification (Energy)\n"
    "6. Stage 5 - Denoising autoencoder + noise-robustness (Energy)\n"
    "7. Stage 6 - Siamese network for anomaly detection (Energy)\n"
    "8. Integrated comparison\n"
    "9. Discussion and conclusions"
)

# ==========================================================================
# 1. Project overview
# ==========================================================================
pdf.add_page()
pdf.section_title("1. Project Overview and Datasets")
pdf.sub_title("1.1 Goal and methodology")
pdf.body_text(
    "The project follows a ten-week syllabus in which each stage builds on artefacts produced "
    "in the previous one. Stages 1 to 3 treat the SCANIA predictive-maintenance dataset; after "
    "Stage 3 the team switched to the Appliance Energy Prediction dataset, which provides "
    "cleaner and denser time series that are better suited to the architectures required by the "
    "later stages (CNN, autoencoder, Siamese). Every stage is implemented in Python with Keras / "
    "TensorFlow, pandas, NumPy, scikit-learn and matplotlib, and documented in a dedicated "
    "Jupyter notebook plus a stage-level report."
)

pdf.sub_title("1.2 Dataset A - SCANIA Component X (Stages 1-3)")
pdf.body_text(
    "SCANIA Component X is a real-world multivariate time series collected from 33,000+ "
    "heavy-duty trucks. The goal is predictive maintenance: predict component failure from "
    "sporadic sensor readouts. Raw data is anonymised (sensor names like 171_0, 666_0, "
    "459_15). We processed train_operational_readouts.csv (~1.2 GB), train_specifications.csv "
    "and train_tte.csv into a single frame keyed on vehicle_id. Because the raw file is large, "
    "the team worked on a 50,000-row sample and used length_of_study_time_step as the "
    "regression target for Stages 2 and 3 (renaming in_study_repair to class_label for the "
    "classification-oriented EDA)."
)

pdf.sub_title("1.3 Dataset B - Appliance Energy Prediction (Stages 4-6)")
pdf.body_text(
    "energydata_complete.csv contains 19,735 observations sampled every 10 minutes over about "
    "4.5 months. Features include indoor temperatures (T1-T9), indoor humidities (RH_1-RH_9), "
    "outdoor weather (T_out, Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint) and two "
    "random reference variables (rv1, rv2). Four cyclical time features are engineered from the "
    "timestamp (hour_sin, hour_cos, dow_sin, dow_cos) for 32 inputs in total. A binary target "
    "HighConsumption is created by thresholding Appliances at its median (60 Wh). The dataset "
    "is split chronologically 70/15/15 into train / validation / test, and windowed into "
    "samples of shape (24, 32) - four hours of history at 10-minute cadence. StandardScaler is "
    "fit on training features only to prevent leakage."
)

# ==========================================================================
# 2. Stage 1
# ==========================================================================
pdf.add_page()
pdf.section_title("2. Stage 1 - EDA and Feature Engineering (SCANIA)")
pdf.body_text(
    "Stage 1 focused on understanding the SCANIA data and producing a clean feature matrix for "
    "downstream modelling:"
)
pdf.bullet("Load and merge operational readouts, vehicle specifications and time-to-event labels "
           "on vehicle_id (sampled to 50,000 rows, ~130 columns).")
pdf.bullet("Timestamp features: lifecycle_progress (0-1 per vehicle), hour_bin, day_bin, "
           "is_weekend, obs_number.")
pdf.bullet("Missing-value strategy: forward-fill within each vehicle -> backward-fill -> "
           "median fill.")
pdf.bullet("Outliers: IQR method with 1.5 x IQR capping per sensor.")
pdf.bullet("Engineered features: lag (1, 2, 3 steps) and rolling means (windows 3, 6, 12) for "
           "the top 3 correlated sensors, categorical time_of_day, and diff (rate-of-change) "
           "features.")
pdf.body_text(
    "Output: processed_data.csv (50,000 rows, 130+ columns) plus seven EDA figures "
    "(correlation, distributions, time series, temporal structure, class distribution, "
    "missing values, outliers)."
)
pdf.add_figure(ass1_fig / "fig1_correlation.png",
               "Figure 1: SCANIA feature correlation heatmap (top sensors).", w=140)
pdf.add_figure(ass1_fig / "fig5_class_dist.png",
               "Figure 2: Binary repair-label distribution (class imbalance).", w=120)

# ==========================================================================
# 3. Stage 2
# ==========================================================================
pdf.add_page()
pdf.section_title("3. Stage 2 - MLP and DNN Regression (SCANIA)")
pdf.body_text(
    "Stage 2 built dense regression baselines on the Stage 1 processed data. The target is "
    "length_of_study_time_step (operational time to event), treated as a continuous value. "
    "Two architectures were trained:"
)
pdf.bullet("MLP: a compact two-hidden-layer network (Dense -> ReLU -> Dropout -> Dense -> "
           "ReLU -> Dense(1, linear)).")
pdf.bullet("DNN: a deeper version with four hidden layers, larger widths and slightly stronger "
           "regularisation.")
pdf.body_text("Both were trained with Adam, MSE loss, EarlyStopping on validation RMSE.")
pdf.sub_title("3.1 Results")
pdf.simple_table(
    headers=["Model", "Split", "RMSE", "MAE"],
    rows=[
        ["MLP", "Train", "59.35", "45.88"],
        ["MLP", "Valid", "60.30", "49.64"],
        ["MLP", "Test",  "71.30", "58.95"],
        ["DNN", "Train", "53.56", "40.21"],
        ["DNN", "Valid", "50.90", "40.72"],
        ["DNN", "Test",  "67.61", "55.18"],
    ],
    col_w=[40, 30, 35, 35],
)
pdf.body_text(
    "The DNN generalised better than the MLP (test RMSE 67.61 vs 71.30, test MAE 55.18 vs "
    "58.95). Both models showed the usual train/test gap caused by per-vehicle sequence "
    "length variability in the 50k sample."
)
pdf.add_figure(ass2_fig / "scatter_actual_vs_predicted.png",
               "Figure 3: DNN predictions vs actual time-to-event (test set).", w=150)

# ==========================================================================
# 4. Stage 3
# ==========================================================================
pdf.add_page()
pdf.section_title("4. Stage 3 - LSTM Regression (SCANIA)")
pdf.body_text(
    "Stage 3 introduced recurrence. Variable-length per-vehicle sequences were built with "
    "padding and a Masking layer, and a single-layer LSTM followed by Dense output was trained "
    "with Adam + MSE and EarlyStopping. A window-length experiment swept the history window "
    "from 6 to 48 steps."
)
pdf.sub_title("4.1 Key results")
pdf.simple_table(
    headers=["Model", "Test RMSE", "Test MAE"],
    rows=[
        ["MLP (Stage 2)", "71.30", "58.95"],
        ["DNN (Stage 2)", "67.61", "55.18"],
        ["LSTM window=12 (saved)", "59.71", "49.92"],
        ["LSTM window=48 (best observed)", "54.04", "44.04"],
    ],
    col_w=[70, 40, 40],
)
pdf.body_text(
    "The saved model (lstm_best.h5) was selected by validation RMSE and used a 12-step window, "
    "improving test RMSE over the Stage 2 DNN by 7.90 and test MAE by 5.25. The lowest test "
    "RMSE observed in the window experiment was at window=48 (RMSE 54.04), confirming that "
    "longer history helps on this dataset, even though the conservative validation-based "
    "selection preferred a shorter window."
)
pdf.body_text(
    "Padding + masking allowed a fair comparison across window lengths while preserving "
    "per-vehicle boundaries."
)

# ==========================================================================
# Transition
# ==========================================================================
pdf.add_page()
pdf.section_title("Dataset transition - Stage 4 onwards")
pdf.body_text(
    "At Stage 4 the team switched to the Appliance Energy Prediction dataset. Two reasons "
    "drove the switch:"
)
pdf.bullet("The SCANIA sample had very short per-vehicle sequences (longest ~16 observations), "
           "which makes 1D CNN / autoencoder / Siamese architectures - all of which benefit "
           "from long, dense windows - awkward to apply.")
pdf.bullet("The Energy dataset provides a uniform 10-minute sampling grid for ~4.5 months, so "
           "rolling 24-step windows with causal Conv1D are natural.")
pdf.body_text(
    "Stages 4, 5 and 6 all share the same chronological 70/15/15 split on this dataset (train "
    "13,790, val 2,936, test 2,937 windows of shape (24, 32)) so their test metrics are "
    "directly comparable."
)

# ==========================================================================
# 5. Stage 4
# ==========================================================================
pdf.section_title("5. Stage 4 - 1D CNN Binary Classification (Energy)")
pdf.body_text(
    "Stage 4 formulated the problem as binary classification: HighConsumption = 1 when "
    "Appliances > 60 Wh (median), 0 otherwise, yielding a near-balanced target (45.6% / "
    "54.4%). A compact 1D CNN was trained with a hyperparameter search over two configurations:"
)
pdf.body_text(
    "    Input(24, 32) -> Conv1D(filters, kernel, causal, ReLU) -> MaxPool1D(2) -> Flatten -> "
    "Dense(64, ReLU) -> Dropout(0.3) -> Dense(1, Sigmoid)"
)
pdf.sub_title("5.1 Results")
pdf.simple_table(
    headers=["filters", "kernel", "Accuracy", "Precision", "Recall", "F1"],
    rows=[
        ["64",  "3", "0.8229", "0.8539", "0.7557", "0.8018"],
        ["128", "5", "0.7960", "0.8015", "0.7572", "0.7787"],
    ],
    col_w=[25, 25, 30, 30, 30, 30],
)
pdf.body_text(
    "Config A (filters=64, kernel=3) was selected. On the test set it scored Accuracy 0.7995, "
    "Precision 0.7687, Recall 0.8498, F1 0.8072, ROC-AUC 0.8851. A logistic regression "
    "baseline on the flattened 24x32 windows (768 features) was competitive (F1 0.8398), "
    "showing that a large share of the class signal is captured by linear combinations of "
    "scaled features - especially the cyclical time features."
)
pdf.add_figure(ass4_fig / "cnn_confusion.png",
               "Figure 4: Stage 4 CNN confusion matrix on the test set.", w=120)

# ==========================================================================
# 6. Stage 5
# ==========================================================================
pdf.add_page()
pdf.section_title("6. Stage 5 - Denoising Autoencoder (Energy)")
pdf.body_text(
    "Stage 5 evaluated the noise-robustness of the Stage 4 classifier and trained a dense "
    "autoencoder to recover the clean signal from noise-corrupted inputs."
)
pdf.sub_title("6.1 Noise simulation and autoencoder")
pdf.body_text(
    "Zero-mean Gaussian noise with sigma = 0.5 (in standardised units) was added to the test "
    "features to simulate sensor malfunction. A fully-connected autoencoder was trained on "
    "clean data only with MSE loss:"
)
pdf.body_text(
    "    Encoder: Input(24 x 32) -> Dense(512) -> Dense(256) -> Dense(128 bottleneck)\n"
    "    Decoder: Dense(256) -> Dense(512) -> Dense(24 x 32, linear)"
)
pdf.sub_title("6.2 Reconstruction quality")
pdf.simple_table(
    headers=["Variant", "MSE vs clean", "Noise reduction"],
    rows=[
        ["AE on clean (baseline)", "0.0970", "-"],
        ["Noisy (before AE)",      "0.2504", "-"],
        ["Denoised (after AE)",    "0.1230", "50.9%"],
    ],
    col_w=[70, 40, 40],
)
pdf.sub_title("6.3 Downstream classifier impact")
pdf.simple_table(
    headers=["Variant", "Accuracy", "Precision", "Recall", "F1"],
    rows=[
        ["Clean",    "0.7995", "0.7687", "0.8498", "0.8072"],
        ["Noisy",    "0.7869", "0.7748", "0.8015", "0.7879"],
        ["Denoised", "0.7692", "0.7620", "0.7746", "0.7683"],
    ],
    col_w=[45, 35, 35, 35, 35],
)
pdf.body_text(
    "Noise alone costs the CNN about 1-2 p.p. of accuracy - the Conv1D receptive field already "
    "averages over local perturbations. The denoising autoencoder halves reconstruction MSE, "
    "but running the classifier on its smooth output actually loses another ~1 p.p. of "
    "accuracy. Pixel-level reconstruction and downstream classification are not the same "
    "objective: the AE's smoothing discards high-frequency detail that the CNN relies on. "
    "This trade-off is a classic finding and motivates joint or task-aware denoising training."
)
pdf.add_figure(ass5_fig / "03_signal_reconstruction.png",
               "Figure 5: Clean vs noisy vs denoised signal for sample windows / channels.",
               w=180)

# ==========================================================================
# 7. Stage 6
# ==========================================================================
pdf.add_page()
pdf.section_title("7. Stage 6 - Siamese Network for Anomaly Detection (Energy)")
pdf.body_text(
    "Stage 6 leaves behind supervised labels at inference time and asks: how far is this window "
    "from anything we have ever seen as normal? A Siamese network is trained with a hand-coded "
    "contrastive loss on balanced pairs."
)
pdf.sub_title("7.1 Pair construction")
pdf.bullet("Positive (Y = 0, similar): two windows with HighConsumption = 0.")
pdf.bullet("Negative (Y = 1, dissimilar): one normal + one high-consumption window.")
pdf.body_text(
    "8,000 pairs per class for training (16,000 total) and 2,000 per class for validation / "
    "test. The Y convention matches Hadsell, Chopra and LeCun (2006)."
)
pdf.sub_title("7.2 Architecture and loss")
pdf.body_text(
    "Shared subnetwork (reuses the Stage 4 backbone plus a second conv block + projection):"
)
pdf.body_text(
    "    Input(24, 32) -> Conv1D(64, k=3, causal) -> MaxPool1D(2) -> Conv1D(128, k=3, causal) "
    "-> MaxPool1D(2) -> Flatten -> Dense(128, ReLU) -> Dropout(0.3) -> Dense(128) -> "
    "L2Normalise  (128-d embedding)"
)
pdf.body_text(
    "    D(a, b) = sqrt( sum_i (a_i - b_i)^2 + epsilon )\n"
    "    L(Y, D) = (1 - Y) * 0.5 * D^2  +  Y * 0.5 * max(0, m - D)^2     (m = 1.0)"
)
pdf.sub_title("7.3 Pair-level evaluation")
pdf.body_text(
    "Validation threshold sweep selected t = 0.2833 (val pair accuracy 0.7395). On test pairs:"
)
pdf.simple_table(
    headers=["Pair metric", "Accuracy", "Precision", "Recall", "F1"],
    rows=[["Siamese", "0.7452", "0.7667", "0.7050", "0.7346"]],
    col_w=[45, 35, 35, 35, 35],
)

pdf.sub_title("7.4 Anomaly detection on individual windows")
pdf.body_text(
    "A reference bank of 2,000 normal embeddings is built from the training set. For each test "
    "window the anomaly score is the minimum Euclidean distance to any reference embedding. "
    "Test ROC-AUC against HighConsumption reaches 0.8685. Thresholded at Youden's J on the "
    "ROC curve: Accuracy 0.786, Precision 0.830, Recall 0.712, F1 0.767."
)
pdf.add_figure(ass6_fig / "roc_and_scores.png",
               "Figure 6: Stage 6 Siamese ROC curve and anomaly-score distribution.", w=180)

# ==========================================================================
# 8. Integrated comparison
# ==========================================================================
pdf.add_page()
pdf.section_title("8. Integrated Comparison")
pdf.sub_title("8.1 SCANIA stages (regression)")
pdf.simple_table(
    headers=["Model", "Test RMSE", "Test MAE"],
    rows=[
        ["MLP (Stage 2)", "71.30", "58.95"],
        ["DNN (Stage 2)", "67.61", "55.18"],
        ["LSTM window=12 saved (Stage 3)", "59.71", "49.92"],
        ["LSTM window=48 best (Stage 3)", "54.04", "44.04"],
    ],
    col_w=[80, 35, 35],
)
pdf.add_figure(root / "figures_final_regression.png",
               "Figure 7: SCANIA regression test error across Stage 2 / 3 models.", w=170)

pdf.sub_title("8.2 Energy stages (classification and anomaly detection)")
rows = []
for _, row in energy_df.iterrows():
    model_label = row["Model"].replace("1D-CNN ", "CNN ")
    auc_val = row["ROC-AUC"]
    auc_str = "-" if pd.isna(auc_val) else f"{auc_val:.4f}"
    rows.append([
        model_label,
        f"{row['Accuracy']:.4f}",
        f"{row['Precision']:.4f}",
        f"{row['Recall']:.4f}",
        f"{row['F1']:.4f}",
        auc_str,
    ])
pdf.simple_table(
    headers=["Model", "Acc", "Prec", "Rec", "F1", "AUC"],
    rows=rows,
    col_w=[70, 22, 22, 22, 22, 22],
)
pdf.add_figure(root / "figures_final_energy.png",
               "Figure 8: Stage 4 / 5 / 6 models on the energy test split.", w=180)

# ==========================================================================
# 9. Discussion
# ==========================================================================
pdf.add_page()
pdf.section_title("9. Discussion")
pdf.sub_title("9.1 Architecture vs task")
pdf.bullet("Forecasting / regression (SCANIA): adding recurrence via an LSTM gave a clear win "
           "over dense MLP / DNN baselines (test RMSE 59.7 vs 67.6 at the saved window, with "
           "54.0 achievable at longer history). Temporal structure matters, and per-vehicle "
           "masking makes it tractable.")
pdf.bullet("Supervised classification (Energy, Stage 4): a compact 1D CNN with causal "
           "convolutions is the strongest closed-set classifier on clean inputs (F1 0.807, "
           "ROC-AUC 0.885). A logistic-regression baseline on flattened windows is "
           "surprisingly competitive (F1 0.84), a reminder that for tabular-style short "
           "windows linear models with cyclical time features capture a large share of the "
           "signal.")
pdf.bullet("Noise robustness and denoising (Stage 5): a fully-connected denoising AE halves "
           "reconstruction MSE under sigma = 0.5 noise, but its smooth output slightly hurts "
           "the downstream CNN. Reconstruction fidelity and downstream discriminability are "
           "not the same objective.")
pdf.bullet("Open-set anomaly detection (Stage 6): a Siamese network trained with contrastive "
           "loss reaches ROC-AUC 0.868 - on par with the supervised CNN (0.885) - using only "
           "a pool of 2,000 normal references at inference time. It offers higher precision "
           "(0.830) at the cost of lower recall (0.712) than the supervised CNN.")

pdf.sub_title("9.2 Engineering lessons")
pdf.bullet("Chronological splits and fitting preprocessing only on train-time data were "
           "critical on both datasets to avoid leakage.")
pdf.bullet("Causal padding kept Conv1D honest about temporal order in Stages 4 and 6.")
pdf.bullet("Reusing the Stage 4 backbone inside the Siamese subnetwork reduced the number of "
           "moving pieces and made the two approaches directly comparable.")
pdf.bullet("Pair-construction convention (Y = 0 similar vs Y = 1 dissimilar) has to match the "
           "contrastive-loss formula signs - the team followed Hadsell, Chopra and LeCun (2006) "
           "to stay consistent with the textbook formulation.")
pdf.bullet("The dataset switch at Stage 4 improved our ability to compare architectures "
           "fairly, at the cost of breaking apples-to-apples comparability with Stages 2 and 3.")

# ==========================================================================
# 10. Conclusions
# ==========================================================================
pdf.section_title("10. Conclusions")
pdf.body_text(
    "The six-stage project delivered a complete deep-learning workflow over time-series data, "
    "covering EDA, dense and recurrent regression, convolutional classification, denoising via "
    "autoencoders, and metric-learning for anomaly detection. On the SCANIA dataset, an LSTM "
    "with per-vehicle masking outperformed dense baselines on regression. After the dataset "
    "switch, on the Appliance Energy data the final Siamese anomaly detector matched the "
    "supervised Stage 4 CNN's ROC-AUC (0.868 vs 0.885) without using any labels at inference "
    "time, using only a small bank of normal reference windows. The three energy-stage models "
    "target different questions - closed-set classification, noise-robust reconstruction, and "
    "open-set anomaly scoring - and together form a complementary monitoring toolkit. All "
    "artefacts (notebooks, saved models, CSVs, figures, per-stage reports) are committed under "
    "the Assigment1 / Assigment2 / Assigment3 / Assignment4 / Assignment5 / Assignment6 "
    "directories in the project repository."
)

pdf.section_title("References")
pdf.body_text(
    "Hadsell, R., Chopra, S., & LeCun, Y. (2006). Dimensionality reduction by learning an "
    "invariant mapping. CVPR.\n"
    "Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of "
    "energy use of appliances in a low-energy house. Energy and Buildings.\n"
    "SCANIA CV AB. SCANIA Component X predictive-maintenance dataset (course-provided "
    "extract)."
)

pdf.output(str(out_path))
print(f"Final report saved to: {out_path}")
