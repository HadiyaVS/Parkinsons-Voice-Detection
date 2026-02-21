# Parkinsons-Voice-Detection

Team name:Vivalavida

Team members
   
    Member1:Hadiya V S-Bharata Mata College,Thrikkakara
   
    Member2:Aesha Suhana P R-Bharata Mata College,Thrikkakara

Hosted project link


Project description

    VoiceCare is a voice-based Parkinson’s detection system that uses machine learning to analyze speech patterns and predict disease risk. It provides a fast, non-invasive screening prototype suitable for telehealth applications.

The problem statement
    
    Parkinson’s disease is a progressive neurological disorder that affects movement and speech, making early diagnosis crucial for effective treatment. Traditional clinical diagnosis methods are often time-consuming, subjective, and expensive, especially in the early stages where symptoms are subtle. There is a need for a low-cost, non-invasive, and accessible screening system that can assist in early detection using voice analysis and machine learning techniques.

The solution

    Our solution is a real-time voice-based Parkinson’s detection prototype where users upload speech recordings for analysis. The system processes the audio, extracts MFCC voice biomarkers, and uses a trained Random Forest classifier to predict disease risk. This approach enables fast, automated, and non-invasive early-stage screening.

Technical Details
  Technologies/components used
      For Software:

       language used:python
       libraries used:pandas,numpy,libosa,seaborn,os,scikit-learn,matplotlib

Features

    Audio Feature Extraction (MFCC):Uses MFCC (Mel-Frequency Cepstral Coefficients) from the librosa library.Extracts 13 coefficients that represent the voice’s frequency characteristics.

    Simulated Parkinson’s Effect:Adds random Gaussian noise to MFCC features.Creates artificial instability in voice patterns.

    Train-Test Split:Uses 80% of data for training.Uses 20% for testing

    Real-Time Audio Prediction (Colab Upload):Lets user upload a new audio file.Extracts MFCC features.Scales them.Predicts Healthy or PD.Shows confidence score

Implementation
 For Software:

    installation
      pip install -r requirements.txt
    Run
      https://802771fba3835f5c2b.gradio.live

Project Documentation
  For Software:

    1.Accuracy matrix & Sample Prediction
    [Accuracy matrix]("C:\Users\unnia\Downloads\WhatsApp Image 2026-02-21 at 9.50.12 AM.jpeg")
    The image shows a machine learning evaluation output with a **model accuracy of about 94.87%** and a displayed **confusion matrix**.
    The matrix indicates:
    * 5 true healthy correctly predicted
    * 2healthy misclassified as Parkinson’s
    * 0 Parkinson’s misclassified as healthy
    * 32 Parkinson’s correctly predicted
    Below the matrix, there is a warning from scikit-learn about feature names and a final prediction stating **“High Risk of Parkinson’s Disease.”*

    2.Prediction Output
    [Prediction Dutput]("C:\Users\unnia\Downloads\WhatsApp Image 2026-02-21 at 9.49.40 AM.jpeg")
    The image shows a dark-themed web app titled **“Parkinson’s Voice Detection.”**
    On the left, a WAV audio file is uploaded with a visible waveform and playback controls, along with **Clear** and **Submit** buttons.
    On the right, the results display **“Low Risk (Healthy)”** with a confidence score, and below it an **MFCC heatmap** visualizing the extracted voice features used for prediction.

    3.Model View
    [Model View]("C:\Users\unnia\Downloads\WhatsApp Image 2026-02-21 at 9.49.23 AM (1).jpeg")
    The image shows a dark-themed web interface for a voice-based health prediction app. It allows users to upload a WAV audio file to determine whether a person is healthy or affected by Parkinson’s.
    On the left side, there is a drag-and-drop area labeled “Drop Audio Here – or – Click to Upload,” along with **Clear** and **Submit** buttons.
    On the right side, there are output panels (labeled “output 0” and “output 1”) where the prediction results and related information would be displayed, plus a **Flag** button below.

Diagrams
 system architecture:

     ┌──────────────────────┐
 │      User (Web UI)   │
 │  Upload WAV File     │
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │  Frontend Interface  │
 │ (Gradio / Web App)   │
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │ Backend Processing   │
 │  - Audio Loading     │
 │  - MFCC Extraction   │
 │  - Feature Scaling   │
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │ ML Model Layer       │
 │ Random Forest Model  │
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │ Prediction Output    │
 │ - Risk Level         │
 │ - Confidence Score   │
 │ - MFCC Heatmap       │
 └──────────────────────┘
  System Overview
    Parkinson’s Voice Detection system follows a **pipeline-based machine learning architecture**, where audio input flows through sequential processing stages until a prediction is produced.It consists of four     main layers:
  1. User Interface Layer
  2. Audio Processing Layer
  3. Machine Learning Layer
  4. Output & Visualization Layer

System Components
  1.User Interface Layer
   Purpose:Accept audio input and display results.
   Component:Web interface (likely built using Gradio)
   Responsibilities:
   upload WAV file
   Show waveform preview
   Display prediction (Healthy / Parkinson’s)
   Show confidence score
   Display MFCC heatmap
This layer interacts directly with the backend when the user clicks **Submit**.

  2.Audio Processing Layer
   A) Audio Loading
      Library Used:librosa
      Function:
      * Loads WAV file
      * Resemples audio
      * Limits duration (e.g., 3 seconds)
      * Converts audio into waveform array

   B) Feature Extraction
     * Extracts MFCC (Mel-Frequency Cepstral Coefficients)**
     * 13 coefficients are computed
     * Mean across time is taken

    Result:
```
Audio → MFCC matrix → 13-length feature vector
```
This converts variable-length audio into fixed numerical input for ML.

---
3.Feature Engineering Layer
  Tool Used:scikit-learn

**Component:**

* `StandardScaler`

**Purpose:**

* Normalize features (mean = 0, std = 1)
* Ensures consistent feature scale
* Prevents model bias

This layer ensures input data matches the scale used during model training.

---

## 4️⃣ Machine Learning Layer

**Model Used:**

* `RandomForestClassifier`

From:

* scikit-learn

**Function:**

* Receives scaled MFCC features
* Passes them through multiple decision trees
* Aggregates predictions
* Outputs:

  * Class label (0 or 1)
  * Probability score

---

## 5️⃣ Output & Visualization Layer

**Outputs:**

* Risk classification:

  * Low Risk (Healthy)
  * High Risk (Parkinson’s)
* Confidence percentage
* MFCC heatmap visualization
* Confusion matrix (during evaluation)

Visualization typically uses:

* `matplotlib`
* `seaborn`

---

# 🔄 3️⃣ Data Flow (Step-by-Step)

### Step 1 — User Upload

User uploads a WAV file via web interface.

⬇

### Step 2 — Audio Preprocessing

* Audio is loaded using librosa
* Trimmed to fixed duration
* Converted to waveform array

⬇

### Step 3 — Feature Extraction

* MFCC features extracted
* Mean aggregation applied
* Output becomes fixed-length feature vector

⬇

### Step 4 — Feature Scaling

* StandardScaler transforms features
* Matches training distribution

⬇

### Step 5 — Model Prediction

* Random Forest predicts class
* `predict_proba()` gives confidence score

⬇

### Step 6 — Result Display

Frontend displays:

* Prediction label
* Confidence %
* MFCC heatmap

---

# 🛠 4️⃣ Tech Stack Interaction

Here’s how technologies interact:

| Layer            | Technology     | Role                     |
| ---------------- | -------------- | ------------------------ |
| UI               | Gradio         | Handles user interaction |
| Audio Processing | librosa        | Extracts MFCC features   |
| ML Framework     | scikit-learn   | Scaling + Model          |
| Model            | Random Forest  | Classification           |
| Visualization    | matplotlib     | Heatmaps & plots         |
| Environment      | Python / Colab | Execution environment    |

### Interaction Flow:

Gradio → calls Python backend → librosa extracts features →
scikit-learn scales → Random Forest predicts →
Result returned to Gradio → Displayed to user.

---

# 🧠 Architectural Style

Your system follows:

### 🔹 Pipeline Architecture

Sequential data transformation:

```
Input → Feature Extraction → Scaling → Model → Output
```

### 🔹 Modular Architecture

Each layer is independent and can be replaced:

* Replace Random Forest with CNN
* Replace MFCC with Spectrogram
* Replace Gradio with FastAPI

---

# 📌 Key Strengths

✔ Simple and modular
✔ Easy to deploy
✔ Fast prediction
✔ Lightweight model

---

# ⚠ Current Limitations

* Uses simulated PD data
* No database storage
* No real-time streaming
* Not clinically validated

---

If you want, I can now give you:

* 🎓 A viva-ready 2-minute explanation
* 📄 A project report version
* 📐 A professional architecture diagram
* 🚀 A deployment architecture (cloud version)


    

     
