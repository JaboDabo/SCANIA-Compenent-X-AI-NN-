"""Generate PDF report for Assignment 2."""
from fpdf import FPDF

class Report(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 8, 'Assignment 2: MLP & DNN Regression - SCANIA Component X', align='C', new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.ln(4)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def sub_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font('Helvetica', '', 10)
        self.cell(0, 5.5, '  - ' + text, new_x="LMARGIN", new_y="NEXT")

    def add_figure(self, path, caption, w=170):
        self.ln(2)
        x = (210 - w) / 2
        self.image(path, x=x, w=w)
        self.ln(2)
        self.set_font('Helvetica', 'I', 9)
        self.cell(0, 5, caption, align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(4)


pdf = Report()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ── Title Page ──
pdf.add_page()
pdf.ln(50)
pdf.set_font('Helvetica', 'B', 22)
pdf.cell(0, 12, 'Assignment 2', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Helvetica', '', 16)
pdf.cell(0, 10, 'Regression: MLP & Deep Neural Network (DNN)', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)
pdf.set_font('Helvetica', '', 13)
pdf.cell(0, 8, 'SCANIA Component X Dataset - Predictive Maintenance', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(20)
pdf.set_font('Helvetica', '', 11)
pdf.cell(0, 7, 'Course: AI & Neural Networks', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, 'University: Astana IT University (AITU)', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, 'Date: March 2026', align='C', new_x="LMARGIN", new_y="NEXT")

# ── 1. Introduction ──
pdf.add_page()
pdf.section_title('1. Introduction')
pdf.body_text(
    'This report presents the implementation and evaluation of two fully connected neural network '
    'architectures - a Multi-Layer Perceptron (MLP) and a Deep Neural Network (DNN) - for a regression '
    'task on the SCANIA Component X dataset. The goal is to predict the time-to-event '
    '(length_of_study_time_step), a continuous variable representing how long a vehicles component '
    'remains in operation before potential failure.'
)
pdf.body_text(
    'The processed dataset from Assignment 1 (50,000 samples, 125 features after preprocessing) '
    'serves as input. We compare the two architectures on RMSE and MAE metrics and draw conclusions '
    'about the applicability of fully connected networks to time series data.'
)

# ── 2. Data Preparation ──
pdf.section_title('2. Data Preparation')
pdf.sub_title('2.1 Dataset')
pdf.body_text(
    'The processed dataset (processed_data.csv) from Assignment 1 was loaded. It contains 50,000 '
    'observations sampled from the full SCANIA dataset (559,091 rows). After excluding non-informative '
    'columns (vehicle_id, time_step, class_label, time_of_day), we retained 125 features including '
    'sensor readings, vehicle specifications, and engineered features (lag, rolling mean, rate of change).'
)

pdf.sub_title('2.2 Target Variable')
pdf.body_text(
    'The regression target is length_of_study_time_step - a continuous value representing the total '
    'operational time of each vehicle in the study. Statistics: Mean = 297.84, Std = 96.67, '
    'Min = 74.0, Max = 510.0.'
)

pdf.sub_title('2.3 Train/Validation/Test Split')
pdf.body_text(
    'The data was split in a 70/15/15 ratio strictly preserving chronological order (no shuffling), '
    'as required for time series data to prevent data leakage from future observations:'
)
pdf.bullet('Train: 35,000 samples (70%)')
pdf.bullet('Validation: 7,500 samples (15%)')
pdf.bullet('Test: 7,500 samples (15%)')
pdf.ln(2)

pdf.sub_title('2.4 Feature Scaling')
pdf.body_text(
    'StandardScaler was applied to all features. Critically, the scaler was fit only on the training '
    'set and then used to transform the validation and test sets, preventing data leakage.'
)

# ── 3. MLP Model ──
pdf.section_title('3. MLP Model (Multi-Layer Perceptron)')
pdf.sub_title('3.1 Architecture')
pdf.body_text(
    'A shallow fully connected network with 3 hidden layers:\n'
    '    Input (125) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)\n'
    'Total parameters: 10,689'
)

pdf.sub_title('3.2 Training Configuration')
pdf.bullet('Optimizer: Adam (learning rate = 0.001)')
pdf.bullet('Loss function: Mean Squared Error (MSE)')
pdf.bullet('Batch size: 64')
pdf.bullet('Max epochs: 100')
pdf.bullet('EarlyStopping: patience=10, restore_best_weights=True')
pdf.ln(2)

pdf.sub_title('3.3 Training Results')
pdf.body_text(
    'The MLP training stopped early at epoch 40 (best weights from epoch 30) with a best '
    'validation loss of 3,173.60. The training loss decreased steadily from ~23,000 to ~2,750, '
    'indicating effective learning.'
)
pdf.add_figure('figures/mlp_loss.png', 'Figure 1: MLP Training & Validation Loss')

# ── 4. DNN Model ──
pdf.section_title('4. Deep Neural Network (DNN)')
pdf.sub_title('4.1 Architecture')
pdf.body_text(
    'A deeper network with 5 hidden layers to test whether increased depth improves performance:\n'
    '    Input (125) -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(16, ReLU) '
    '-> Dense(8, ReLU) -> Dense(1, Linear)\n'
    'Total parameters: 27,137'
)

pdf.sub_title('4.2 Training Results')
pdf.body_text(
    'The DNN stopped at epoch 34 (best weights from epoch 24) with a best validation loss of '
    '2,361.80 - significantly better than the MLPs 3,173.60. The deeper architecture converged '
    'to a lower loss faster despite having more parameters.'
)
pdf.add_figure('figures/dnn_loss_comparison.png',
               'Figure 2: DNN Loss (left) and MLP vs DNN Validation Loss Comparison (right)', w=180)

# ── 5. Quality Evaluation ──
pdf.add_page()
pdf.section_title('5. Quality Evaluation')
pdf.sub_title('5.1 Metrics Comparison')
pdf.body_text(
    'RMSE and MAE were computed on all three splits for both models:'
)

# Metrics table
pdf.set_font('Helvetica', 'B', 10)
col_w = [30, 35, 35, 35]
headers = ['Model', 'Split', 'RMSE', 'MAE']
for i, h in enumerate(headers):
    pdf.cell(col_w[i], 7, h, border=1, align='C')
pdf.ln()

pdf.set_font('Helvetica', '', 10)
data = [
    ('MLP', 'Train', '53.69', '40.02'),
    ('MLP', 'Validation', '56.33', '45.31'),
    ('MLP', 'Test', '68.82', '55.89'),
    ('DNN', 'Train', '52.28', '38.32'),
    ('DNN', 'Validation', '48.60', '38.53'),
    ('DNN', 'Test', '65.93', '53.62'),
]
for row in data:
    for i, val in enumerate(row):
        pdf.cell(col_w[i], 7, val, border=1, align='C')
    pdf.ln()
pdf.ln(4)

pdf.body_text(
    'The DNN outperforms the MLP on all splits. On the test set, the DNN achieves an RMSE of 65.93 '
    'versus 68.82 for the MLP - a 4.2% improvement. The MAE shows a similar pattern (53.62 vs 55.89).'
)

pdf.body_text(
    'Note on MSE scale: The raw MSE values appear large (2,000-6,000) because the target variable '
    'ranges from 74 to 510. An RMSE of ~66 relative to a standard deviation of ~97 indicates the '
    'models explain a meaningful portion of the target variance.'
)

pdf.sub_title('5.2 Actual vs Predicted - Scatter Plot')
pdf.add_figure('figures/scatter_actual_vs_predicted.png',
               'Figure 3: Actual vs Predicted on Test Set (MLP left, DNN right)', w=180)

pdf.sub_title('5.3 Actual vs Predicted - Line Plot')
pdf.add_figure('figures/line_actual_vs_predicted.png',
               'Figure 4: Actual vs Predicted for first 200 test samples (MLP top, DNN bottom)', w=180)

# ── 6. Model Comparison ──
pdf.add_page()
pdf.section_title('6. Model Comparison & Discussion')

pdf.sub_title('MLP (3 hidden layers, 10,689 params)')
pdf.bullet('Faster training (~0.6ms/step)')
pdf.bullet('Stopped at epoch 40, best at epoch 30')
pdf.bullet('Higher test error (RMSE: 68.82)')
pdf.bullet('More stable but potentially underfitting')
pdf.ln(2)

pdf.sub_title('DNN (5 hidden layers, 27,137 params)')
pdf.bullet('Slightly slower training (~1ms/step)')
pdf.bullet('Stopped at epoch 34, best at epoch 24')
pdf.bullet('Lower test error (RMSE: 65.93)')
pdf.bullet('Greater capacity captured more complex patterns')
pdf.ln(2)

pdf.sub_title('Impact of Depth')
pdf.body_text(
    'The deeper DNN architecture provided a modest but consistent improvement over the MLP. '
    'The additional layers (128 and 8 neurons) allowed the network to learn more complex '
    'non-linear mappings between sensor readings and time-to-event. However, the improvement '
    'is not dramatic (~4%), suggesting that the relationship captured by the shallower network '
    'already accounts for most learnable patterns in the tabular features.'
)

# ── 7. Conclusions ──
pdf.section_title('7. Conclusions')
pdf.body_text(
    '1. Both MLP and DNN successfully learned to predict time-to-event from sensor data, with '
    'the DNN achieving slightly better performance (Test RMSE: 65.93 vs 68.82).'
)
pdf.body_text(
    '2. EarlyStopping with restore_best_weights was essential - both models showed signs of '
    'overfitting (validation loss oscillating while training loss decreased), and early stopping '
    'prevented degradation.'
)
pdf.body_text(
    '3. Chronological splitting (no shuffling) is critical for time series to avoid data leakage. '
    'The higher test error compared to validation error suggests the model encounters distribution '
    'shift in later time periods, which is expected in temporal data.'
)
pdf.body_text(
    '4. Fully connected networks treat each observation independently and cannot model temporal '
    'dependencies between consecutive readings. For improved performance on this time series task, '
    'recurrent architectures (RNN, LSTM) or Transformers would be more appropriate - these will '
    'be explored in subsequent assignments.'
)
pdf.body_text(
    '5. The best model (DNN) was saved as mlp_best.h5 for use in future assignments.'
)

# Save
pdf.output('Report_Assignment2.pdf')
print('Report saved as Report_Assignment2.pdf')
