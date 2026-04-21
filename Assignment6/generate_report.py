"""Generate PDF report for Assignment 6 using metrics and figures from the notebook run."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF

here = Path(__file__).parent
fig_dir = here / "figures"
fig_dir.mkdir(exist_ok=True)

comparison_df = pd.read_csv(here / "model_comparison.csv")

fig, ax = plt.subplots(figsize=(9, 4.5))
plot_df = comparison_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]]
plot_df.plot(kind="bar", ax=ax, rot=25, ylim=(0, 1))
ax.set_title("Stage 4 / 5 / 6 Models on the Energy Dataset Test Split")
ax.set_ylabel("Score")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(fig_dir / "model_comparison.png", dpi=140)
plt.close()

siamese_row = comparison_df[comparison_df["Model"].str.startswith("Siamese")].iloc[0]
cnn_row = comparison_df[comparison_df["Model"].str.startswith("1D-CNN classifier")].iloc[0]
cnn_noisy = comparison_df[comparison_df["Model"].str.contains("noisy")].iloc[0]
cnn_denoised = comparison_df[comparison_df["Model"].str.contains("denoised")].iloc[0]


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, "Assignment 6: Siamese Network for Anomaly Detection - Appliance Energy",
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

# Title page
pdf.add_page()
pdf.ln(50)
pdf.set_font("Helvetica", "B", 22)
pdf.cell(0, 12, "Assignment 6", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 16)
pdf.cell(0, 10, "Siamese Network for Anomaly Detection", align="C", new_x="LMARGIN", new_y="NEXT")
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
    "This report describes Stage 6 of the project: building a Siamese neural network that learns a "
    "metric over 24-step consumption windows so that windows with normal consumption cluster "
    "together and windows with high consumption are pushed apart. The learned distance is then used "
    "as an anomaly score - a test window is flagged as anomalous when its distance to the nearest "
    "reference normal window exceeds a tuned threshold."
)
pdf.body_text(
    "The Siamese network reuses the same 1D CNN backbone from Stage 4, with the final classification "
    "head replaced by a 128-dimensional embedding projection. The two embeddings are compared by a "
    "Lambda Euclidean-distance layer, and the whole network is trained end-to-end with a contrastive "
    "loss implemented from scratch."
)
pdf.body_text(
    "In the final section the Siamese detector is compared against the supervised Stage 4 CNN and "
    "the Stage 5 denoising autoencoder + CNN pipeline on the same chronological test split."
)

# 2. Data and pair construction
pdf.section_title("2. Data Preparation and Pair Construction")
pdf.sub_title("2.1 Dataset and splits")
pdf.body_text(
    "The dataset is energydata_complete.csv (19,735 rows x 29 columns). The binary target "
    "HighConsumption was created in Stage 4 by thresholding Appliances at its median (60 Wh). "
    "The data is split in a 70/15/15 chronological ratio and rescaled with StandardScaler fit on "
    "the training set only. Rolling windows of 24 steps (4 hours of history) are built, producing "
    "samples of shape (24, 32):"
)
pdf.bullet("Train:      (13790, 24, 32)")
pdf.bullet("Validation: (2936, 24, 32)")
pdf.bullet("Test:       (2937, 24, 32)")

pdf.sub_title("2.2 Pair generation")
pdf.body_text(
    "Following the assignment brief, training pairs are built in two balanced groups:"
)
pdf.bullet("Positive pair (Y = 0, similar): two windows where HighConsumption = 0.")
pdf.bullet("Negative pair (Y = 1, dissimilar): one normal window + one high-consumption window.")
pdf.body_text(
    "The (Y = 0 -> similar, Y = 1 -> dissimilar) convention follows Hadsell, Chopra and LeCun "
    "(2006), which is also the form of contrastive loss used below. 8,000 pairs per class were "
    "drawn for training (16,000 total), and 2,000 per class for validation and test."
)

# 3. Architecture
pdf.section_title("3. Siamese Architecture")
pdf.sub_title("3.1 Shared 1D CNN subnetwork")
pdf.body_text(
    "The shared subnetwork reuses the Stage 4 CNN backbone, dropping the sigmoid classification "
    "head and projecting into a 128-dimensional L2-normalised embedding:"
)
pdf.body_text(
    "    Input(24, 32) -> Conv1D(64, k=3, causal, ReLU) -> MaxPool1D(2) -> "
    "Conv1D(128, k=3, causal, ReLU) -> MaxPool1D(2) -> Flatten -> Dense(128, ReLU) -> "
    "Dropout(0.3) -> Dense(128) -> L2Normalise"
)
pdf.body_text(
    "Normalising the embeddings keeps Euclidean distances in a bounded range, which stabilises "
    "contrastive training."
)

pdf.sub_title("3.2 Distance layer and contrastive loss")
pdf.body_text(
    "The Siamese model takes two inputs that pass through the shared subnetwork. A Lambda layer "
    "computes the Euclidean distance between the two embeddings:"
)
pdf.body_text("    D(a, b) = sqrt( sum_i (a_i - b_i)^2 + epsilon )")
pdf.body_text("Contrastive loss is implemented from scratch:")
pdf.body_text("    L(Y, D) = (1 - Y) * 0.5 * D^2  +  Y * 0.5 * max(0, m - D)^2")
pdf.body_text(
    "with margin m = 1.0. The (1 - Y) branch pulls similar pairs together, and the Y branch "
    "pushes dissimilar pairs past the margin. Training: Adam (lr 1e-3), batch size 128, up to "
    "15 epochs, EarlyStopping (patience 4) and ReduceLROnPlateau callbacks on val_loss."
)
pdf.add_figure(fig_dir / "siamese_training_curves.png",
               "Figure 1: Contrastive loss and pair accuracy across epochs.", w=180)

# 4. Threshold tuning and pair-level evaluation
pdf.add_page()
pdf.section_title("4. Threshold Tuning and Pair Evaluation")
pdf.body_text(
    "Pairs are classified dissimilar when D > t. The threshold was swept across the validation "
    "distance range and the point maximising validation pair accuracy was retained:"
)
pdf.bullet("Best validation threshold: 0.2833")
pdf.bullet("Validation pair accuracy: 0.7395")
pdf.body_text("Applying the same threshold on the balanced test pair set yields:")

pdf.set_font("Helvetica", "B", 10)
col_w = [45, 35, 35, 35, 35]
headers = ["Metric", "Accuracy", "Precision", "Recall", "F1"]
for i, h in enumerate(headers):
    pdf.cell(col_w[i], 7, h, border=1, align="C")
pdf.ln()
pdf.set_font("Helvetica", "", 10)
row_vals = ["Test pairs", "0.7452", "0.7667", "0.7050", "0.7346"]
for i, v in enumerate(row_vals):
    pdf.cell(col_w[i], 7, v, border=1, align="C")
pdf.ln(10)

pdf.add_figure(fig_dir / "threshold_tuning.png",
               "Figure 2: Validation distance distribution and pair accuracy vs threshold.", w=180)

# 5. Anomaly detection
pdf.add_page()
pdf.section_title("5. Anomaly Detection on Individual Test Windows")
pdf.body_text(
    "For open-set anomaly detection the trained subnetwork is used as a feature extractor. A "
    "bank of 2,000 normal reference embeddings is built from the training set (HighConsumption = 0). "
    "For each test window the anomaly score is the minimum Euclidean distance to any reference "
    "embedding. Windows whose nearest normal neighbour is far away are flagged as anomalous."
)
pdf.bullet(f"Anomaly score (test set): mean 0.2007, std 0.1740")
pdf.bullet(f"ROC-AUC on HighConsumption target: {siamese_row['ROC-AUC']:.4f}")
pdf.body_text(
    "Thresholding the anomaly score at the Youden-J optimum on the ROC curve gives a single "
    "operating point for downstream reporting:"
)
pdf.set_font("Helvetica", "B", 10)
col_w = [45, 35, 35, 35, 35]
headers = ["At Youden's J", "Accuracy", "Precision", "Recall", "F1"]
for i, h in enumerate(headers):
    pdf.cell(col_w[i], 7, h, border=1, align="C")
pdf.ln()
pdf.set_font("Helvetica", "", 10)
row_vals = [
    "Siamese",
    f"{siamese_row['Accuracy']:.4f}",
    f"{siamese_row['Precision']:.4f}",
    f"{siamese_row['Recall']:.4f}",
    f"{siamese_row['F1']:.4f}",
]
for i, v in enumerate(row_vals):
    pdf.cell(col_w[i], 7, v, border=1, align="C")
pdf.ln(10)
pdf.add_figure(fig_dir / "roc_and_scores.png",
               "Figure 3: ROC curve and anomaly score distribution on the test set.", w=180)
pdf.add_figure(fig_dir / "top_anomalies.png",
               "Figure 4: Five test windows with the highest anomaly score (Appliances channel).",
               w=160)

# 6. Cross-stage comparison
pdf.add_page()
pdf.section_title("6. Integrated Model Comparison")
pdf.body_text(
    "Stage 4, 5, and 6 models are compared on the identical chronological test split (2,937 "
    "windows). Stage 5 metrics are taken verbatim from Assignment5.ipynb; the Siamese row is "
    "computed from the same notebook run as above."
)

pdf.set_font("Helvetica", "B", 9)
col_w = [70, 22, 22, 22, 22, 22]
headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
for i, h in enumerate(headers):
    pdf.cell(col_w[i], 7, h, border=1, align="C")
pdf.ln()
pdf.set_font("Helvetica", "", 9)
for _, row in comparison_df.iterrows():
    model_label = row["Model"].replace("1D-CNN ", "CNN ")
    auc_val = row["ROC-AUC"]
    auc_str = "-" if pd.isna(auc_val) else f"{auc_val:.4f}"
    vals = [
        model_label,
        f"{row['Accuracy']:.4f}",
        f"{row['Precision']:.4f}",
        f"{row['Recall']:.4f}",
        f"{row['F1']:.4f}",
        auc_str,
    ]
    for i, v in enumerate(vals):
        pdf.cell(col_w[i], 7, v, border=1, align="C")
    pdf.ln()
pdf.ln(4)

pdf.add_figure(fig_dir / "model_comparison.png",
               "Figure 5: Side-by-side accuracy, precision, recall and F1 across Stage 4/5/6 models.",
               w=180)

# 7. Discussion
pdf.add_page()
pdf.section_title("7. Discussion")
pdf.body_text(
    "Supervised classification (Stage 4 CNN) is the strongest closed-set model on clean inputs "
    f"(Accuracy {cnn_row['Accuracy']:.3f}, F1 {cnn_row['F1']:.3f}, ROC-AUC {cnn_row['ROC-AUC']:.3f}). "
    "Causal 1D convolutions over 24-step windows capture the shape of consumption spikes better "
    "than a flat feature vector."
)
pdf.body_text(
    "Robustness to noise (Stage 5). Adding sigma = 0.5 Gaussian noise to the standardised features "
    f"degrades the CNN only mildly (Accuracy {cnn_noisy['Accuracy']:.3f}, F1 {cnn_noisy['F1']:.3f}) - "
    "the Conv1D receptive field already averages over local perturbations. The denoising autoencoder "
    "reduces reconstruction MSE from 0.250 to 0.123 (-50.9%), but running the classifier on the "
    "reconstructed test set actually loses another ~1 p.p. of accuracy "
    f"({cnn_denoised['Accuracy']:.3f}). The AE optimises pixel-level fidelity, not downstream "
    "discriminability - a classic denoising / classification trade-off."
)
pdf.body_text(
    "Anomaly detection (Stage 6). The Siamese network is not a classifier: it scores each window "
    "by distance to the nearest known-normal reference embedding. Contrastive training pulls "
    "normals together and pushes anomalies past the margin, so the distance is a well-calibrated "
    f"anomaly score. Its test ROC-AUC of {siamese_row['ROC-AUC']:.3f} is competitive with the "
    f"supervised CNN's {cnn_row['ROC-AUC']:.3f} even though the Siamese never sees labels at "
    f"inference - only a pool of 2,000 normal references. Thresholded at Youden's J it reaches "
    f"Accuracy {siamese_row['Accuracy']:.3f} / F1 {siamese_row['F1']:.3f}, trading recall "
    f"({siamese_row['Recall']:.3f}) for higher precision ({siamese_row['Precision']:.3f}) "
    "compared to the CNN."
)
pdf.body_text(
    "Each model answers a different question. The CNN says is this window a high-consumption "
    "spike? given full supervision; the denoising AE says what did this sensor stream look like "
    "before the noise? given only a clean prior; and the Siamese says how far is this window "
    "from anything we have ever seen as normal? given only a bank of normal references. A "
    "production monitoring pipeline benefits from combining all three."
)

# 8. Conclusions
pdf.section_title("8. Conclusions")
pdf.body_text(
    "1. A Siamese network built on the Stage 4 CNN backbone with a Lambda Euclidean distance "
    "layer and a hand-coded contrastive loss was trained end-to-end to separate normal and "
    "high-consumption windows in embedding space."
)
pdf.body_text(
    "2. On balanced test pairs the network reaches pair Accuracy = 0.7452 / F1 = 0.7346 at the "
    "validation-tuned threshold 0.283."
)
pdf.body_text(
    f"3. Used as an open-set anomaly detector via nearest-reference distance, the Siamese "
    f"network achieves test ROC-AUC = {siamese_row['ROC-AUC']:.4f}, on par with the supervised "
    f"Stage 4 CNN (ROC-AUC = {cnn_row['ROC-AUC']:.4f})."
)
pdf.body_text(
    "4. The denoising autoencoder from Stage 5 halves reconstruction MSE under noise but its "
    "smoothed output slightly hurts the downstream CNN, showing that denoising quality and "
    "classification quality are not the same objective."
)
pdf.body_text(
    "5. Artifacts saved: siamese_embedding.keras, siamese_model.keras, model_comparison.csv, and "
    "the figures used in this report."
)

pdf.output(str(here / "Report_Assignment6.pdf"))
print("Report saved as Report_Assignment6.pdf")
