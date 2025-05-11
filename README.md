# Remote Sensing Image Change Captioning with BLIP

This project utilizes the **BLIP (Bootstrapped Language-Image Pretraining)** model to generate descriptive captions that explain differences between two remote sensing (RS) images. The core objective is to fine-tune the BLIP model using the **LEVIR-CC** dataset to produce accurate and meaningful change descriptions.

---

## 📁 Project Structure

```
.
├── Datasets
│   ├── levir-cc           # Extracted LEVIR-CC dataset
│   └── data_new           # Empty folder for new processed data
├── saved_models           # Directory to save trained models
├── levircc.py             # Script for dataset conversion
├── fine_tune.py           # Training and testing script
├── requirements.txt       # Required libraries
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Python Environment
Ensure Python **3.11.1** is installed.

### 2. Install Required Libraries
```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Preparation

1. **Download and Extract Dataset**  
   Download the LEVIR-CC dataset and extract it into a folder named:
   ```
   Datasets/levir-cc
   ```

2. **Create Necessary Folders**
   ```bash
   mkdir -p Datasets/data_new
   mkdir saved_models
   ```

3. **Convert Dataset**  
   Convert the dataset into `.hdf5` and `.json` formats:
   ```bash
   python levircc.py
   ```

---

## 🚀 Training and Testing

Once the dataset is prepared, start training and testing with the following command:

```bash
python fine_tune.py
```

---

## 📬 Contact

For issues or contributions, please open an issue or pull request on the project repository.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).




This project aims to leverage the BLIP (Bootstrapped Language-Image Pretraining) model to identify and describe differences between two remote sensing (RS) images in the form of captions. The primary focus is fine-tuning the BLIP model for this task to achieve accurate and meaningful descriptions of changes.

Current Progress
* Fine-tuned the BLIP model for single-image captioning in the domain of remote sensing images.
* Future steps include fine-tuning and adapting the model for change detection captioning, where the differences between two RS images are described in natural language.
