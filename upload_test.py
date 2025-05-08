import os
from server_api import file_upload

# 업로드할 폴더 경로 정의
camera_folders = {
    "CT00010": r"C:\Users\user\Desktop\yolo\captures_01\20250325\08\bus",    # camera_01
    # "CT-04020": r"C:\Users\user\Desktop\yolo\captures_02\20250325\08\bus"   # camera_02
}

# 이미지 업로드
for unitid, folder_path in camera_folders.items():
    print(f"\n📂 [{unitid}] 지점으로 업로드 시작: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")
        continue

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(folder_path, filename)
            print(f"  ⬆ 업로드 중 → [{unitid}] {filename}")
            file_upload(unitid, file_path)
