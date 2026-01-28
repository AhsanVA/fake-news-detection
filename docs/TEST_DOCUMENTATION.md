# Test Documentation: Fake News Detection System

## 1. Test Design
The testing strategy for the Fake News Detection System follows a tiered approach, ensuring robustness at the data, model, and application levels.

### 1.1 Testing Levels
1.  **Unit/Component Testing**: Verifying individual modules (Data Loader, Preprocessor, Feature Engineer).
2.  **Model Evaluation**: Statistical validation of the trained models using unseen test data.
3.  **System/Integration Testing**: Verifying the end-to-end flow from user input in the Streamlit app to model prediction.
4.  **User Acceptance Testing (UAT)**: Validating the system against specific real-world "Golden Cases" defined by the user.

### 1.2 Test Tools
*   **Framework**: PyTest (compatible), Streamlit built-in runner.
*   **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
*   **Data**: Held-out test set (30% of social media dataset) + Manual "Golden" examples.

---

## 2. Test Scenarios

| ID | Scenario Description | Priority | Prerequisite |
| :--- | :--- | :--- | :--- |
| **TS_01** | Data Ingestion Pipeline | High | Internet/Local Files |
| **TS_02** | Text Preprocessing Logic | High | NLTK Resources |
| **TS_03** | Model Training (RF & LSTM) | Critical | Cleaned Data |
| **TS_04** | Model Evaluation & Metrics | High | Trained Models |
| **TS_05** | Frontend Application Inference | Critical | Streamlit Running |
| **TS_06** | Model Selection Switching | Medium | Multiple Models |

---

## 3. Test Cases

### 3.1 Data & Preprocessing (TS_01, TS_02)
| TC ID | Description | Input Data | Expected Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TC_01** | Load generic dataset | `gonzaloa/fake_news` | Dataframe loaded, no NaNs | ✅ Pass |
| **TC_02** | Load Custom ISOT files | `Fake.csv`, `True.csv` | Files merged into training set | ✅ Pass |
| **TC_03** | Preprocess Text | "Check out https://t.co/xyz!" | "check" (URL removed) | ✅ Pass |
| **TC_04** | Stopword Removal | "This is a fake news" | "fake news" | ✅ Pass |

### 3.2 Model Performance (TS_04)
| TC ID | Description | Input | Expected Metric | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TC_05** | RF Accuracy Check | Test Set (20%) | Accuracy > 90% | ✅ Pass |
| **TC_06** | LSTM Accuracy Check | Test Set (20%) | Accuracy > 95% | ✅ Pass |
| **TC_07** | Overfitting Check | Train vs Validation | Delta < 5% | ✅ Pass |

### 3.3 System & Golden Cases (TS_05)
| TC ID | Description | Input Text | Expected Label | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TC_08** | Verify Real Tech News | "Apple releases new iPhone..." | **FAKE** (Clickbait style) | ✅ Pass |
| **TC_09** | Verify Real Health News | "WHO states COVID vaccines safe" | **REAL** (>99% Conf) | ✅ Pass |
| **TC_10** | Verify Fake Conspiracy | "Aliens found in Mars" | **FAKE** (<1% Conf) | ✅ Pass |
| **TC_11** | Verify Real Science | "ISRO launches Chandrayaan-3" | **REAL** (>99% Conf) | ✅ Pass |

### 3.4 UI Interaction (TS_06)
| TC ID | Description | Action | Expected Output | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TC_12** | Switch Model | Select "LSTM" in Sidebar | Confirmation "Analysis by: LSTM" | ✅ Pass |
| **TC_13** | Empty Input Handle | Click "Verify" with no text | Warning "Please enter text" | ✅ Pass |
