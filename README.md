# **Interactive-Automatic-Image-Labeling-Platform-Development**

This repository is an extended version of [ScribblePrompt](https://huggingface.co/spaces/halleewong/ScribblePrompt), providing an **interactive image labeling platform** with advanced features like auto-inference, 3D segmentation, and batch processing.

---

## **Features**
- Interactive annotation with points, bounding boxes, and scribbles.
- Automatic labeling with guide-free inference.
- Support for 3D images (e.g., NIfTI) and videos.
- Batch processing with progress tracking.
- Post-processing mask editor for refinement.
- Model fine-tuning using custom training data.

---

## **Environment Setup**

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

---

## **Usage**

### **1. Interactive Annotation**
- Use bounding boxes, points, or scribbles to annotate images interactively.
- Adjust the mask in the post-processing editor.

### **2. Automatic Labeling**
- Enable "Auto-Inference Mode" for guide-free labeling.

### **3. 3D Image and Video Segmentation**
- Upload NIfTI files or videos.
- Use sliders to select slices or frames for segmentation.

### **4. Batch Processing**
- Upload multiple images, select from the dropdown menu, and track progress using the "Done" button.

---

## **Project Structure**

```plaintext
Interactive-Automatic-Image-Labeling-Platform-Development/
├── checkpoints/           # Pretrained models directory
├── test_examples/         # Example images and test data
├── app.py                 # Main application script
├── network.py             # Network definitions (e.g., UNet, SAM)
├── predictor.py           # Model inference logic
├── requirements.txt       # Python dependencies
├── LICENSE                # License file
└── README.md              # Project documentation
```

---

## **Contributing**

1. Fork the repository.
2. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a Pull Request.

---

## **License**

This project is licensed under the [MIT License](LICENSE).

---

## **Acknowledgments**

This project builds on the [ScribblePrompt](https://huggingface.co/spaces/halleewong/ScribblePrompt) repository, extending its functionality and usability.
