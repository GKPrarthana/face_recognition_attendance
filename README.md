# Face Recognition Based Attendance System
A deep learning project to recognize faces from pre-uploaded images and log attendance.

## Setup
1. Clone the repo: `git clone https://github.com/GKPrarthana/face_recognition_attendance.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Add raw images to `dataset/` (not tracked in Git).
4. Run preprocessing: `python src/preprocess.py`
5. Build face database: `python src/build_database.py`

## Progress
- [x] Step 1: Define problem and scope
- [x] Step 2: Gather and preprocess dataset
- [x] Step 3: Choose face recognition approach (DeepFace with embeddings)
- [ ] Step 4: Build the project (recognition and logging)