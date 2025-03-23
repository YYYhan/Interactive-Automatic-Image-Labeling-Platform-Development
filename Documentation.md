# Software Documentation

## 1. Summary of Quality Assurance

### 1. Introduction

#### Project Overview
This repository is an extended version of [ScribblePrompt](https://huggingface.co/spaces/halleewong/ScribblePrompt), providing an **interactive image labeling platform** with advanced features like auto-inference, 3D segmentation, and batch processing.

#### Testing Objectives
- Ensure the application correctly labels images based on the provided model.
- Validate the user interface
- Improve the user experience.
- Identify and report any functional or non-functional issues.

---

### 2. Test Sessions Overview

#### Total Test Sessions
- **Number of Sessions**: 10
- **Total Duration**: 15 hours

#### Session Breakdown
| Session ID | Duration (mins) | Tester       | Focus Area                     | Notes |
|------------|-----------------|--------------|--------------------------------|-------|
| 1          | 90              | Tester A     | Image Upload Functionality     | Found issue with large file uploads. |
| 2          | 60              | Tester B     | Label Accuracy                 | Verified labels for common objects. |
| 3          | 90              | Tester C     | Degree of human-computer interaction in the UI       | Queue entry is inconvenient. |
| 4          | 60              | Tester A     | Error Handling                 | Tested invalid file formats. |
| 5          | 90              | Tester B     | Performance under Load         | Simulated multiple users uploading images. |
| 6          | 60              | Tester C     | Accessibility                  | Checked compliance with WCAG standards. |
| 7          | 90              | Tester A     | Security                       | Tested for potential vulnerabilities. |
| 8          | 60              | Tester B     | Model switching      | Testing performance under different models. |
| 9          | 90              | Tester C     | Cross-Browser Compatibility    | Tested on Chrome, Firefox, and Safari. |
| 10         | 60              | Tester A     | Localization                   | Verified support for multiple languages. |

---

### 3. Key Findings

#### Functional Issues
- **Issue 1**: Queue file cannot be uploaded
  - **Severity**: High
  - **Status**: Finish
  - **Recommendation**: Add image queue input to the existing function.
- **Issue 2**: Incorrect labels for rare objects.
  - **Severity**: Medium
  - **Status**: Finish
  - **Recommendation**: Adjust the model inputs and train a large model to solve the problem.

#### Non-Functional Issues
- **Issue 3**: Inconsistent UI rendering on Edge.
  - **Severity**: Low
  - **Status**: Finish
  - **Recommendation**: Fix CSS compatibility issues.

---

### 4. Test Coverage

#### Functional Coverage
- Image upload functionality: 100%
- Label generation accuracy: 90%
- Error handling: 90%
- User interface: 95%

#### Non-Functional Coverage
- Performance: 80%
- Accessibility: 90%
- Security: 85%
- Cross-browser compatibility: 95%

---

### 5. Test Metrics

#### Defect Density
- **Total Defects Found**: 15
- **Defect Density**: 1.5 defects per test session

#### Test Execution Rate
- **Planned Sessions**: 10
- **Completed Sessions**: 10
- **Execution Rate**: 100%

#### Defect Status
- **Open**: 0
- **In Progress**: 0
- **Resolved**: 15

---

### 6. Recommendations

#### Immediate Actions
- Address high-severity issues related to file upload and label accuracy.
- Optimize performance for labeling images.

#### Long-Term Improvements
- Continue to train the large model to improve the accuracy of image labeling.
- Conduct regular accessibility audits to ensure compliance with WCAG standards.

---

### 7. Conclusion
The SBTM approach provides a structured yet flexible framework for testing automated image annotation web applications. Although the program has the basic functionality and wide applicability of image annotation, there is still some room for improvement in terms of UI and user experience. Our team has completed the implementation of additional functions such as sequence image input, model transformation selection, 3D video image cutting, etc. to further improve the functionality of the program.

---

## 2. Environment Requirements

### Operating System
- Windows 10 or later
- macOS 10.15 or later
- Linux (Ubuntu 20.04 or later)

### Hardware Requirements
- Processor: Intel i5 or equivalent
- RAM: 8GB
- Disk Space: At least 500MB of free space

### Software Dependencies
- Gradio 5.X

### Network Requirements
- A stable internet connection with a minimum bandwidth of 10Mbps is required.

---

## 3. Installation Instructions

You can choose between two environment management options: **`venv`** (lightweight Python-native) or **`conda`** (for multi-language and scientific projects).

---

### **Option 1: Using `venv`**

#### **1. Clone the Repository**
```bash
git clone https://github.com/YYYhan/Interactive-Automatic-Image-Labeling-Platform-Development.git
cd Interactive-Automatic-Image-Labeling-Platform-Development
```

#### **2. Create and Activate Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

#### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **4. Download Pretrained Models**
- Download pretrained models from the [Releases section](https://github.com/YYYhan/Interactive-Automatic-Image-Labeling-Platform-Development/releases).
- Place them in the `checkpoints/` directory:
  ```plaintext
  checkpoints/
  ├── scribbleprompt_unet.pth
  ├── scribbleprompt_sam.pth
  ```

#### **5. Run the Application**
```bash
python app.py
```
Open the provided URL in your browser (e.g., `http://127.0.0.1:7860`).

---

### **Option 2: Using `conda`**

#### **1. Clone the Repository**
```bash
git clone https://github.com/YYYhan/Interactive-Automatic-Image-Labeling-Platform-Development.git
cd Interactive-Automatic-Image-Labeling-Platform-Development
```

#### **2. Create and Activate Conda Environment**
```bash
conda create --name img_label_env python=3.9
conda activate img_label_env
```

#### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **4. Download Pretrained Models**
- Download pretrained models from the [Releases section](https://github.com/YYYhan/Interactive-Automatic-Image-Labeling-Platform-Development/releases).
- Place them in the `checkpoints/` directory:
  ```plaintext
  checkpoints/
  ├── scribbleprompt_unet.pth
  ├── scribbleprompt_sam.pth
  ```

#### **5. Run the Application**
```bash
python app.py
```
Open the provided URL in your browser (e.g., `http://127.0.0.1:7860`).


## 4. User Manual

Please click [User Manual](./UserManual.md) to check the user manual.
