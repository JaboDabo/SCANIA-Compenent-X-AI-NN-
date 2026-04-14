"""Generate PDF report for Assignment 4 using metrics and figures from the notebook run."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF

here = Path(__file__).parent
fig_dir = here / "figures"
fig_dir.mkdir(exist_ok=True)

# Rename extracted notebook figures to stable names (only if present).
renames = {
    "cell_3_1.png": "class_balance.png",
    "cell_8_2.png": "cnn_training.png",
    "cell_9_3.png": "cnn_confusion.png",
}
for src, dst in renames.items():
    s = fig_dir / src
    if s.exists():
        s.rename(fig_dir / dst)

# Metrics from notebook run.
cnn_m = dict(Accuracy=0.7995, Precision=0.7687, Recall=0.8498, F1=0.8072)
base_m = dict(Accuracy=0.8383, Precision=0.8223, Recall=0.8580, F1=0.8398)

cnn_search = pd.DataFrame([
    {"filters": 64,  "kernel_size": 3, "Accuracy": 0.8229, "Precision": 0.8539, "Recall": 0.7557, "F1": 0.8018},
    {"filters": 128, "kernel_size": 5, "Accuracy": 0.7960, "Precision": 0.8015, "Recall": 0.7572, "F1": 0.7787},
])

# Build comparison bar chart from scratch.
comp = pd.DataFrame([{"Model": "1D CNN", **cnn_m}, {"Model": "Logistic Regression", **base_m}]).set_index("Model")
fig, ax = plt.subplots(figsize=(8, 4))
comp.plot(kind="bar", ax=ax, rot=0, ylim=(0, 1))
ax.set_title("1D CNN vs Logistic Regression (Test Metrics)")
ax.set_ylabel("Score")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(fig_dir / "model_comparison.png", dpi=120)
plt.close()


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, "Assignment 4: 1D CNN Binary Classification - Appliance Energy",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, t):
        self.set_font("Helvetica", "B", 14); self.ln(4)
        self.cell(0, 10, t, new_x="LMARGIN", new_y="NEXT"); self.ln(2)

    def sub_title(self, t):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, t, new_x="LMARGIN", new_y="NEXT"); self.ln(1)

    def body_text(self, t):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, t); self.ln(2)

    def bullet(self, t):
        self.set_font("Helvetica", "", 10)
        self.cell(0, 5.5, "  - " + t, new_x="LMARGIN", new_y="NEXT")

    def add_figure(self, path, caption, w=170):
        self.ln(2)
        x = (210 - w) / 2
        self.image(str(path), x=x, w=w)
        self.ln(2)
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)


pdf = Report()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# Title
pdf.add_page()
pdf.ln(50)
pdf.set_font("Helvetica", "B", 22)
pdf.cell(0, 12, "Assignment 4", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 16)
pdf.cell(0, 10, "Binary Classification with 1D CNN", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)
pdf.set_font("Helvetica", "", 13)
pdf.cell(0, 8, "Appliance Energy Prediction Dataset", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(20)
pdf.set_font("Helvetica", "", 11)
pdf.cell(0, 7, "Course: AI & Neural Networks", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "University: Astana IT University (AITU)", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Date: April 2026", align="C", new_x="LMARGIN", new_y="NEXT")

# 1. Introduction
pdf.add_page()
pdf.section_title("1. Introduction")
pdf.body_text(
    "This report presents the implementation and evaluation of a 1D Convolutional Neural Network "
    "(1D CNN) for a binary classification task on the Appliance Energy Prediction dataset. "
    "The dataset contains 19,735 observations sampled every 10 minutes over about 4.5 months, "
    "covering temperature and humidity readings across nine rooms plus outdoor weather variables."
)
pdf.body_text(
    "The target variable is created by thresholding appliance energy consumption at its median "
    "value: HighConsumption = 1 when Appliances > median (60 Wh), otherwise 0. The CNN is "
    "compared against a logistic regression baseline trained on the same rolling-window "
    "representation."
)

# 2. Data Preparation
pdf.section_title("2. Data Preparation")
pdf.sub_title("2.1 Dataset")
pdf.body_text(
    "The dataset energydata_complete.csv contains 19,735 rows and 29 columns. Features include "
    "indoor temperatures (T1-T9), indoor humidities (RH_1-RH_9), outdoor weather (T_out, "
    "Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint), and two random reference variables "
    "(rv1, rv2). Four cyclical time features were engineered from the timestamp: hour_sin, "
    "hour_cos, dow_sin, dow_cos. Total input features: 32."
)

pdf.sub_title("2.2 Target Variable")
pdf.body_text(
    "The binary target HighConsumption was created using the median appliance consumption "
    "(60 Wh). The class distribution is close to balanced, with 45.6% high-consumption and "
    "54.4% low/normal samples. Because the imbalance gap is well below 20%, class weighting "
    "was not applied during training."
)
pdf.add_figure(fig_dir / "class_balance.png", "Figure 1: Target Class Distribution", w=120)

pdf.sub_title("2.3 Chronological Split and Windowing")
pdf.body_text(
    "The data was split in a 70/15/15 ratio preserving chronological order. Rolling windows of "
    "24 consecutive time steps (4 hours of history at 10-minute intervals) were constructed so "
    "that each sample has shape (24, 32). StandardScaler was fit on the training set only and "
    "applied to validation and test sets to prevent data leakage."
)
pdf.bullet("Train sequences: (13790, 24, 32)")
pdf.bullet("Validation sequences: (2936, 24, 32)")
pdf.bullet("Test sequences: (2937, 24, 32)")
pdf.ln(2)

# 3. CNN Model
pdf.section_title("3. 1D CNN Model")
pdf.sub_title("3.1 Architecture")
pdf.body_text(
    "A compact 1D CNN was used, treating each 24-step window as a 1D signal over 32 channels:\n"
    "    Input (24, 32) -> Conv1D(filters, kernel, causal, ReLU) -> MaxPool1D(2) -> Flatten "
    "-> Dense(64, ReLU) -> Dropout(0.3) -> Dense(1, Sigmoid)"
)
pdf.body_text(
    "Causal padding ensures the convolution at position t only uses steps <= t, keeping the "
    "temporal order intact. Binary cross-entropy loss and the Adam optimizer "
    "(learning rate 1e-3) were used, with EarlyStopping (patience 4 on val_loss, "
    "restore_best_weights=True) and ModelCheckpoint saving the best epoch. Training used "
    "batch size 64 with a maximum of 20 epochs."
)

pdf.sub_title("3.2 Hyperparameter Search")
pdf.body_text("Two CNN configurations were compared on the validation set:")
pdf.set_font("Helvetica", "B", 10)
col_w = [25, 30, 30, 30, 30, 30]
headers = ["filters", "kernel", "Accuracy", "Precision", "Recall", "F1"]
for i, h in enumerate(headers):
    pdf.cell(col_w[i], 7, h, border=1, align="C")
pdf.ln()
pdf.set_font("Helvetica", "", 10)
for _, row in cnn_search.iterrows():
    vals = [str(int(row["filters"])), str(int(row["kernel_size"])),
            f"{row['Accuracy']:.4f}", f"{row['Precision']:.4f}",
            f"{row['Recall']:.4f}", f"{row['F1']:.4f}"]
    for i, v in enumerate(vals):
        pdf.cell(col_w[i], 7, v, border=1, align="C")
    pdf.ln()
pdf.ln(4)
pdf.body_text(
    "Config A (filters=64, kernel=3) achieved the higher validation F1 (0.802 vs 0.779) and "
    "was selected as the best model. It was saved to cnn_classifier.h5."
)

pdf.sub_title("3.3 Training Curves")
pdf.add_figure(fig_dir / "cnn_training.png",
               "Figure 2: CNN Training vs Validation Loss and Accuracy", w=180)

# 4. Evaluation
pdf.add_page()
pdf.section_title("4. Evaluation")
pdf.sub_title("4.1 Test Metrics")
pdf.set_font("Helvetica", "B", 10)
col_w = [55, 35, 35, 35, 35]
headers = ["Model", "Accuracy", "Precision", "Recall", "F1"]
for i, h in enumerate(headers):
    pdf.cell(col_w[i], 7, h, border=1, align="C")
pdf.ln()

pdf.set_font("Helvetica", "", 10)
rows = [("1D CNN (best)", cnn_m), ("Logistic Regression", base_m)]
for name, m in rows:
    vals = [name, f"{m['Accuracy']:.4f}", f"{m['Precision']:.4f}",
            f"{m['Recall']:.4f}", f"{m['F1']:.4f}"]
    for i, v in enumerate(vals):
        pdf.cell(col_w[i], 7, v, border=1, align="C")
    pdf.ln()
pdf.ln(4)

pdf.add_figure(fig_dir / "model_comparison.png",
               "Figure 3: CNN vs Logistic Regression (Test Metrics)", w=170)

pdf.sub_title("4.2 Confusion Matrix")
pdf.add_figure(fig_dir / "cnn_confusion.png",
               "Figure 4: CNN Confusion Matrix on Test Set", w=120)

# 5. Discussion
pdf.add_page()
pdf.section_title("5. Discussion")
pdf.body_text(
    "The 1D CNN reached reasonable test performance (Accuracy 0.80, F1 0.81). The smaller "
    "config (filters=64, kernel=3) generalized better than the larger one (filters=128, "
    "kernel=5), suggesting that a wider receptive field and more filters caused mild "
    "overfitting on this dataset of modest size."
)
pdf.body_text(
    "The logistic regression baseline, fit on the flattened 24x32 window (768 features), is "
    "competitive with - and in this run slightly stronger than - the CNN (F1 0.840 vs 0.807). "
    "This suggests that a large part of the signal separating high vs normal consumption is "
    "captured by simple linear combinations of the scaled features, especially the cyclical "
    "time features (hour_sin, hour_cos) which correlate strongly with daily usage patterns."
)
pdf.body_text(
    "The confusion matrix shows the CNN favors recall over precision on the High class "
    "(recall 0.85 vs precision 0.77): it catches most high-consumption windows but also "
    "misclassifies a noticeable share of normal windows as high. A threshold above 0.5 "
    "would trade recall for precision if false positives are more costly."
)
pdf.body_text(
    "Causal padding was important here: without it the model would have access to future "
    "time steps within a window. Keeping the split strictly chronological and fitting the "
    "scaler only on the training set prevents the model from peeking into the future."
)

pdf.section_title("6. Conclusions")
pdf.body_text(
    "1. A binary classification task was successfully formulated from the raw Appliance Energy "
    "dataset by thresholding at the median consumption (60 Wh)."
)
pdf.body_text(
    "2. A 1D CNN with causal Conv1D layers was trained on 24-step windows of 32 features and "
    "achieved Accuracy = 0.7995 and F1 = 0.8072 on the held-out test set."
)
pdf.body_text(
    "3. A logistic regression baseline on the same flattened windows reached F1 = 0.8398, "
    "showing that the dataset has a strong linear component and that a heavier CNN is not "
    "automatically better for tabular-style short windows."
)
pdf.body_text(
    "4. For further gains on this problem, richer temporal architectures (stacked Conv1D, "
    "dilated convolutions, LSTM/GRU, Transformers) and class-balanced thresholds beyond 0.5 "
    "would be natural next steps."
)
pdf.body_text(
    "5. The best CNN model was saved as cnn_classifier.h5 in the Assignment4 directory."
)

pdf.output(str(here / "Report_Assignment4.pdf"))
print("Report saved as Report_Assignment4.pdf")
