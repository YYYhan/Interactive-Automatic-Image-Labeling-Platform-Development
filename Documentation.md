# Software Documentation

## 1. Summary of Quality Assurance

### Testing Strategy
We have implemented the following testing strategies to ensure the quality of the software:
- **Unit Testing**: Each module is tested independently to ensure that individual functions and classes work as expected.
- **Integration Testing**: Ensures that the interfaces between different modules work correctly, especially for database and API integrations.
- **System Testing**: End-to-end testing is performed to simulate real user workflows and ensure the system functions as a whole.

### Testing Tools
The following tools were used for testing:
- **JUnit**: For unit testing of Java code.
- **Selenium**: For automated testing of the web interface.
- **Postman**: For API testing.

### Test Results
- Unit test coverage: 85%
- Integration test pass rate: 100%
- System test pass rate: 95%

### Quality Assurance Process
- Code reviews are conducted by team members before each commit.
- Continuous integration is implemented using Jenkins to automatically run tests after each commit.

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
