"""Generate PDF report for Assignment 5 (denoising autoencoder, energy dataset)."""
from pathlib import Path

from fpdf import FPDF

here = Path(__file__).parent
img_dir = here / "images"


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8,
                  "Assignment 5: Denoising Autoencoder - Appliance Energy",
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
        self.set_font("Helvetica", "B", 10)
        for i, h in enumerate(headers):
            self.cell(col_w[i], 7, h, border=1, align="C")
        self.ln()
        self.set_font("Helvetica", "", 10)
        for row in rows:
            for i, v in enumerate(row):
                self.cell(col_w[i], 7, str(v), border=1, align="C")
            self.ln()
        self.ln(3)


pdf = Report()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ---- Title page ----------------------------------------------------------
pdf.add_page()
pdf.ln(50)
pdf.set_font("Helvetica", "B", 22)
pdf.cell(0, 12, "Assignment 5", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 16)
pdf.cell(0, 10, "Denoising Autoencoder and Feature Extraction",
         align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)
pdf.set_font("Helvetica", "", 13)
pdf.cell(0, 8, "Appliance Energy Prediction Dataset",
         align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(20)
pdf.set_font("Helvetica", "", 11)
pdf.cell(0, 7, "Course: AI & Neural Networks", align="C",
         new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "University: Astana IT University (AITU)", align="C",
         new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Date: April 2026", align="C",
         new_x="LMARGIN", new_y="NEXT")

# ---- 1. Introduction -----------------------------------------------------
pdf.add_page()
pdf.section_title("1. Introduction")
pdf.body_text(
    "This report summarises Stage 5 of the course project on the Appliance Energy Prediction "
    "dataset, building directly on the chronological splits and 24-step windows produced in "
    "Assignment 4. The stage has two goals: first, quantify how sensitive the Assignment 4 "
    "1D CNN classifier is to sensor-like noise, and second, train a fully-connected "
    "autoencoder on clean data and evaluate how well it can denoise corrupted windows and "
    "recover downstream classification performance."
)
pdf.body_text(
    "The assignment was implemented in a single Jupyter notebook (Assignment5.ipynb). All "
    "preprocessing, autoencoder training, reconstruction and classifier evaluation steps "
    "completed successfully, and the trained autoencoder was saved as autoencoder.h5."
)

# ---- 2. Experimental Setup ----------------------------------------------
pdf.section_title("2. Experimental Setup")
pdf.sub_title("2.1 Dataset and classification baseline")
pdf.body_text(
    "The experiment reuses the chronological, standardised sequences from Assignment 4 built "
    "from energydata_complete.csv (19,735 rows x 29 columns, sampled every 10 minutes). The "
    "target HighConsumption = 1 when the Appliances column exceeds its median (60 Wh), 0 "
    "otherwise. The data is split 70/15/15 chronologically and windowed into samples of shape "
    "(24, 32). The saved Assignment 4 model cnn_classifier.h5 is used as the fixed downstream "
    "classifier; on the clean test set it scores Accuracy 0.7995 and F1 0.8072."
)

pdf.sub_title("2.2 Noise simulation")
pdf.body_text(
    "To simulate sensor malfunctions, Gaussian noise with zero mean and sigma = 0.5 (in "
    "standardised units) was added to every feature of the test set. Because the inputs were "
    "scaled to unit variance, sigma = 0.5 corresponds to half a standard deviation per "
    "feature, which is a significant but realistic perturbation for instrumented sensor "
    "streams."
)
pdf.add_figure(img_dir / "01_noise_simulation.png",
               "Figure 1: Clean vs noisy signal for several features / sample windows.",
               w=180)

# ---- 3. Autoencoder -----------------------------------------------------
pdf.add_page()
pdf.section_title("3. Autoencoder Denoising Approach")
pdf.sub_title("3.1 Architecture")
pdf.body_text(
    "A symmetric fully-connected autoencoder maps the 768-dimensional flattened windows "
    "(24 steps x 32 features) through a compressed bottleneck and back:"
)
pdf.body_text(
    "    Encoder: Input(768) -> Dense(512, ReLU) -> Dense(256, ReLU) -> Dense(128, ReLU)\n"
    "    Decoder: Dense(256, ReLU) -> Dense(512, ReLU) -> Dense(768, linear)"
)
pdf.body_text(
    "The 128-unit bottleneck compresses the input by exactly 6x, forcing the network to "
    "capture a compact representation of the healthy signal manifold."
)

pdf.sub_title("3.2 Training")
pdf.body_text(
    "The autoencoder is trained on the training set only, with input == output == clean "
    "window - so the model never sees noisy data during training. Loss: Mean Squared Error. "
    "Optimiser: Adam. EarlyStopping was used with patience 10 epochs; training ran for 31 "
    "epochs and the best validation loss was 0.0568."
)
pdf.add_figure(img_dir / "02_training_curves.png",
               "Figure 2: Autoencoder training and validation MSE across epochs.", w=150)

# ---- 4. Results --------------------------------------------------------
pdf.section_title("4. Results and Evaluation")
pdf.sub_title("4.1 Signal reconstruction quality")
pdf.body_text(
    "The noisy test data (which the autoencoder had never seen) was passed through the "
    "trained model. Reconstruction quality was measured as MSE against the original clean "
    "signal:"
)
pdf.simple_table(
    headers=["Variant", "MSE vs clean", "Noise reduction"],
    rows=[
        ["Autoencoder on clean (baseline)", "0.0970", "-"],
        ["Noisy (before AE)",               "0.2504", "-"],
        ["Denoised (after AE)",             "0.1230", "50.9%"],
    ],
    col_w=[70, 40, 40],
)
pdf.add_figure(img_dir / "03_signal_reconstruction.png",
               "Figure 3: Clean / noisy / denoised signal for sample windows and channels.",
               w=180)

pdf.sub_title("4.2 Impact on downstream classification")
pdf.body_text(
    "The Assignment 4 CNN classifier was run on three variants of the test set: clean, noisy "
    "(sigma = 0.5), and autoencoder-reconstructed."
)
pdf.simple_table(
    headers=["Variant", "Accuracy", "Precision", "Recall", "F1"],
    rows=[
        ["Clean (baseline)",    "0.7995", "0.7687", "0.8498", "0.8072"],
        ["Noisy (sigma=0.5)",   "0.7869", "0.7748", "0.8015", "0.7879"],
        ["Denoised by AE",      "0.7692", "0.7620", "0.7746", "0.7683"],
    ],
    col_w=[50, 32, 32, 32, 32],
)
pdf.add_figure(img_dir / "04_classifier_metrics.png",
               "Figure 4: CNN performance on clean, noisy and denoised test sets.", w=180)
pdf.add_figure(img_dir / "05_confusion_matrices.png",
               "Figure 5: Confusion matrices for all three test-set variants.", w=180)

# ---- 5. Discussion / Conclusions ----------------------------------------
pdf.add_page()
pdf.section_title("5. Discussion")
pdf.body_text(
    "Sensitivity to noise. Adding sigma = 0.5 Gaussian noise caused a measurable but modest "
    "drop in the CNN's performance (F1 0.807 -> 0.788). The Conv1D receptive field already "
    "averages over local perturbations, which explains why the CNN is not catastrophically "
    "sensitive to per-step noise at this level."
)
pdf.body_text(
    "Autoencoder reconstruction. The denoising autoencoder cut reconstruction MSE from 0.250 "
    "to 0.123 (a 50.9% reduction), recovering clearly cleaner signal envelopes in the "
    "reconstruction plots."
)
pdf.body_text(
    "Reconstruction vs discriminability trade-off. Despite the clear MSE improvement, running "
    "the classifier on the denoised test set actually loses about another percentage point of "
    "accuracy (0.787 -> 0.769). Pushing the signal through the 128-unit bottleneck "
    "destroys some of the high-frequency detail the CNN relies on to separate high and low "
    "consumption windows. Reconstruction fidelity (pixel-level MSE) and downstream "
    "discriminability are not the same objective. To recover the lost accuracy on denoised "
    "inputs, the CNN would need to be retrained or fine-tuned on autoencoder reconstructions, "
    "or the AE could be trained jointly with the classifier using a task-aware loss."
)

pdf.section_title("6. Conclusions")
pdf.body_text(
    "1. A dense autoencoder with a 128-unit bottleneck was successfully trained on clean "
    "24-step windows of the energy dataset and used to denoise noise-corrupted test inputs."
)
pdf.body_text(
    "2. The autoencoder reduced reconstruction MSE from 0.250 to 0.123 (-50.9%), confirming "
    "that it captures the low-frequency structure of the energy signals."
)
pdf.body_text(
    "3. Passing the noisy (sigma = 0.5) test set through the Assignment 4 CNN drops F1 from "
    "0.807 to 0.788; passing the autoencoder-reconstructed set further drops F1 to 0.768, "
    "illustrating the classic denoising / classification trade-off."
)
pdf.body_text(
    "4. Autoencoders remain a valuable preprocessing tool for sensor-like noise, but for best "
    "downstream classification performance the classifier should be retrained or fine-tuned on "
    "reconstructions, or the two networks should be trained jointly with a task-aware loss."
)
pdf.body_text(
    "5. Deliverables: Assignment5.ipynb, autoencoder.h5, the five figures in the images/ "
    "directory, and this report."
)

pdf.output(str(here / "Report_Assignment5.pdf"))
print("Report saved as Report_Assignment5.pdf")
