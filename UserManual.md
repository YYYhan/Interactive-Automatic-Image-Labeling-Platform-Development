# User Manual

## 1. Introduction

### Overview
This repository is an extended version of [ScribblePrompt](https://huggingface.co/spaces/halleewong/ScribblePrompt), providing an **interactive image labeling platform** with advanced features like auto-inference, 3D segmentation, and batch processing.


### Target Audience
This manual is intended for:
- End-users who will interact with the software.
- Administrators responsible for setting up and maintaining the software.

---

## 2. Features

- Interactive annotation with points, bounding boxes, and scribbles.
- Automatic labeling with guide-free inference.
- Support for 3D images (e.g., NIfTI) and videos.
- Batch processing with progress tracking.
- Post-processing mask editor for refinement.
- Model fine-tuning using custom training data.

### Screenshots

![Initial Page](images/Initial.png)  
- Initial page loaded by the user.

![Instructions](images/Instructions.png)  
- Instructions of function.

![ModelSelection](images/ModelSelection.png)  
- ModelSelection.


![Scribble&Click](images/Scribble&Click.png)  
- Scribble&Click.

![Prediction](images/Prediction.png)  
- Prediction


![ClicksMode](images/ClicksMode.png)  
- ClicksMode.

![BoundingBox](images/BoundingBox.png)  
- BoundingBox.


![MaskEditor](images/MaskEditor.png)  
- MaskEditor.

![Queue](images/Queue.png)  
- Queue.

---

## 3. Getting Started

### System Requirements
- **Operating System**: Windows 10 or Ubuntu 20.04.
- **RAM**: 8GB.
- **Disk Space**: 500MB of free space.
- **Software Dependencies**: Gradio 5.X .

### Installation Steps
1. Download the project from the [github website](https://github.com/YYYhan/Interactive-Automatic-Image-Labeling-Platform-Development/tree/main).
2. Configure the required environment according to README.md.
3. Launch the website.

### First Run
- During the first run, you need to configure the system environment and the required extensions, and download the developed model.

---

## 4. Step-by-Step Guide

### Image Upload 
1. Click on the Image List box to select the images that need to be uploaded.
2. Select the desired box selection mode.
3. Interact with images through a mouse.(Detailed operation demonstration videos can be viewed)
4. Click Apply Scribble.



## 5. Troubleshooting

### Common Issues
- **Unable to Update Image**:
  - Make sure your network and server are in good working order.
  - Make sure that the project download is complete and the environment is configured correctly.
- **Image Upload Fails**:
  - Ensure that local images to be uploaded are correctly formatted and complete.
  - Ensure that the file size does not exceed the limit (e.g., 100MB).

### Error Messages
- **Error 404: Page Not Found**:
  - Ensure that you are accessing the correct URL.
  - Refresh the page or clear your browser cache.
- **Error 500: Internal Server Error**:
  - Contact the support team for assistance.

---

## 6. FAQs

### Q1: How do I reset the state of image?

- Click on Clear All Input and re-enter the image.

### Q2: How do I achieve recognition of 3D video images?
- The software supports CSV and Excel files (.csv, .xlsx).

### Q3: How do I contact support?

- You can contact support via email at scyjz21@nottingham.edu.cn.

---

## 7. Contact Information

### Support Channels
- **Email**: scyjz21@nottingham.edu.cn


### Feedback
- We welcome your feedback! Please send your suggestions or report issues to scyjz21@nottingham.edu.cn.

---

## 8. Appendix

### Glossary
- **API**: Application Programming Interface, a set of protocols for building software.

### Keyboard Shortcuts
- `Ctrl + S`: Save the current file.

### Additional Resources
- [Official Documentation](./Documentation.md)
- [Tutorial Videos](Video/Recording_Software_Demonstration.mp4)