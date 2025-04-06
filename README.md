# Face Recognition Based Attendance System
A deep learning project to recognize faces in real-time via webcam and log attendance.

## Setup
1. Clone the repo: `git clone https://github.com/GKPrarthana/face_recognition_attendance.git`
2. Create a virtual environment: `python -m venv face_env`
3. Activate it:
   - Windows: `face_env\Scripts\activate`
   - macOS/Linux: `source face_env/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run preprocessing: `python src/preprocess.py`
6. Build face database: `python src/build_database.py`
7. Start real-time recognition: `python src/recognize_and_log.py`

## Progress
- [x] Step 1: Define problem and scope
- [x] Step 2: Gather and preprocess dataset
- [x] Step 3: Choose face recognition approach (DeepFace with embeddings)
- [x] Step 4: Build the project (real-time recognition and logging)
- [ ] Step 5: Test and optimize