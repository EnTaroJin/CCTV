import cv2
import datetime
import os
import time
import numpy as np
import shutil
from send2trash import send2trash  # send2trash 모듈 추가
import platform
import subprocess
import server_api


tracked_objects = set()  # 추적된 객체를 저장하기 위한 집합
tracked_objects.add(None) 
processed_files = set()  # 이미 처리된 파일을 기록할 집합
camera_number = 1

# 비디오 저장 중지 신호를 위한 플래그
stop_flag = False

def is_video_file_stable(filepath, check_interval=1.0, checks=3):
    """파일 크기를 여러 번 측정해서 일정 시간 동안 변화 없으면 안정된 파일로 간주"""
    try:
        previous_size = os.path.getsize(filepath)
        for _ in range(checks):
            time.sleep(check_interval)
            current_size = os.path.getsize(filepath)
            if current_size != previous_size:
                return False
            previous_size = current_size
        return True
    except Exception:
        return False

def reconnect(cap, url):
    cap.release()
    cap = cv2.VideoCapture(url)
    return cap


def zoom_in(image, zoom_factor=1.0):
    if zoom_factor <= 1.0:
        return image

    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    new_width = int(image.shape[1] / zoom_factor)
    new_height = int(image.shape[0] / zoom_factor)
    left = max(center_x - new_width // 2, 0)
    right = min(center_x + new_width // 2, image.shape[1])
    top = max(center_y - new_height // 2, 0)
    bottom = min(center_y + new_height // 2, image.shape[0])
    
    cropped_image = image[top:bottom, left:right]
    zoomed_image = cv2.resize(cropped_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return zoomed_image


# def preprocess_image(image):
#         # 히스토그램 평활화
#     yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
#     yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
#     image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
#     # 감마 보정
#     gamma = 1.5
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     image = cv2.LUT(image, table)
    
#     return image


def empty_trash():
    if platform.system() == "Windows":
        subprocess.run(["powershell", "-command", "Clear-RecycleBin -Force"])
    elif platform.system() == "Darwin":
        subprocess.run(["osascript", "-e", "tell app \"Finder\" to empty the trash"])
    elif platform.system() == "Linux":
        home = os.path.expanduser("~")
        trash_dir = os.path.join(home, ".local/share/Trash/files")
        shutil.rmtree(trash_dir, ignore_errors=True)


def save_capture(image, obj_type, direction, previous_count, current_count, model, counter, classes_to_count, results=None):
    if current_count != previous_count:
        print(f"{obj_type} {direction} 값이 변경되었습니다. 이전 값: {previous_count}, 현재 값: {current_count}")

        current_date = datetime.datetime.now().strftime("%Y%m%d")
        current_hour = datetime.datetime.now().strftime("%H")
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_path = os.path.join("captures", current_date, current_hour, obj_type)
        os.makedirs(folder_path, exist_ok=True)

        base_image_name = os.path.join(folder_path, f"{current_time}.jpg")
        image_name = base_image_name
        index = 1
        while os.path.exists(image_name):
            image_name = f"{base_image_name}_{index}.jpg"
            index += 1

        # ✅ YOLO 결과 재사용
        if results is not None:
            result_image = counter.start_counting(image, results)
        else:
            tracks = model.track(image, persist=True, show=False, classes=classes_to_count)
            result_image = counter.start_counting(image, tracks)

        cv2.imwrite(image_name, result_image)
        print(f"캡처 저장: {image_name}")



def save_capture2(image, obj_type, model, counter, classes_to_count, results=None):
    global camera_number

    current_date = datetime.datetime.now().strftime("%Y%m%d")
    current_hour = datetime.datetime.now().strftime("%H")
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(f"captures_0{camera_number}", current_date, current_hour, obj_type)
    os.makedirs(folder_path, exist_ok=True)

    base_image_name = os.path.join(folder_path, f"{current_time}.jpg")
    image_name = base_image_name
    index = 1
    while os.path.exists(image_name):
        image_name = f"{base_image_name}_{index}.jpg"
        index += 1

    # ✅ YOLO 결과 재사용
    if results is not None:
        result_image = counter.start_counting(image, results)
    else:
        tracks = model.track(image, persist=True, show=False, classes=classes_to_count)
        result_image = counter.start_counting(image, tracks)

    cv2.imwrite(image_name, result_image)
    print(f"캡처 저장: {image_name}")
    server_api.file_upload("CT00010", image_name)



def stop_saving_video():
    global stop_flag
    stop_flag = True


def save_video_in_chunks(url, base_output_folder, duration=180, fourcc_str='mp4v', zoom_factor=1.0, c_number=2):
    global stop_flag
    global camera_number
    camera_number = c_number
    frame_skip = 1

    cap = cv2.VideoCapture(url)
    assert cap.isOpened(), "비디오 스트림에 연결할 수 없습니다."

    while cap.isOpened():
        # 폴더 및 파일 경로 설정
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        current_time_folder = datetime.datetime.now().strftime("%H")
        current_time_video = datetime.datetime.now().strftime("%H%M%S")
        output_folder = os.path.join(base_output_folder, current_date, current_time_folder)
        os.makedirs(output_folder, exist_ok=True)

        # FPS 확인 및 보정
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 60:
            print("[경고] FPS 값 비정상. 기본값 15.0 사용")
            fps = 15.0

        # 해상도 및 코덱 설정
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        video_filename = os.path.join(output_folder, f"video_{current_time_video}.mp4")
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

        start_time = time.time()
        frame_counter = 0
        frames_written = 0  # ✅ 실제 저장된 프레임 수 체크용

        while time.time() - start_time < duration:
            if stop_flag:
                break

            success, frame = cap.read()
            if not success:
                print("프레임을 읽는 데 실패했습니다. 다시 연결을 시도합니다.")
                cap = reconnect(cap, url)
                success, frame = cap.read()
                if not success:
                    print("재연결 실패. 2초 후 다시 시도합니다.")
                    time.sleep(2)
                    continue

            # frame = zoom_in(frame, zoom_factor)  # 줌 기능 필요 시 활성화
            if frame_counter % frame_skip == 0:
                video_writer.write(frame)
                frames_written += 1  # ✅ 프레임 저장 수 증가

            frame_counter += 1

        video_writer.release()

        # ✅ 프레임이 하나도 저장되지 않았으면 파일 삭제
        if frames_written == 0 and os.path.exists(video_filename):
            print(f"[경고] 저장된 프레임이 없습니다. 손상된 파일을 삭제합니다: {video_filename}")
            os.remove(video_filename)

        if stop_flag:
            break

    cap.release()



def delete_old_folders(base_folder, days_to_keep=3):
    current_date = datetime.datetime.now()
    threshold_date = current_date - datetime.timedelta(days=days_to_keep)
    
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            folder_date = datetime.datetime.strptime(folder, "%Y%m%d")
            if folder_date < threshold_date:
                print(f"Deleting old folder: {folder_path}")
                shutil.rmtree(folder_path)
                empty_trash()  # 휴지통 비우기
    

def record_processed_file(file_name):
    current_date = datetime.datetime.now().strftime("%Y%m%d")  # 날짜 형식
    absolute_path = os.path.abspath(file_name)
    current_time = datetime.datetime.now().strftime("%H:%M:%S")

    # 날짜별 로그 폴더 생성
    log_folder = os.path.join("영상 처리 로그", current_date)
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, "processed_files.txt")

    with open(log_path, "a") as f:
        f.write(f"{absolute_path},{current_date},{current_time}\n")


def load_processed_files():
    processed_files = set()
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    log_path = os.path.join("영상 처리 로그", current_date, "processed_files.txt")

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 1:
                    file_name = parts[0]
                    processed_files.add(os.path.abspath(file_name))

    return processed_files



def process_video_files(base_folder, model, counter, classes_to_count):
    while True:
        current_time = datetime.datetime.now()
        current_hour = current_time.hour

        # 현재 시간이 5시 이상이면 비디오 파일 처리
        if 5 <= current_hour < 17:
            delete_old_folders(base_folder)  # 오래된 폴더 삭제
            current_date = current_time.strftime("%Y%m%d")

            # 현재시간부터 17시까지의 폴더를 처리
            for hour in range(current_hour, 18):
                hour_folder = os.path.join(base_folder, current_date, f"{hour:02}")
                if os.path.exists(hour_folder):
                    process_videos_in_folder(hour_folder, model, counter, classes_to_count)

            # 다음 날로 넘어감
            current_date = (current_time + datetime.timedelta(days=1)).strftime("%Y%m%d")
            for hour in range(0, 5):  # 다음 날 0시부터 5시까지 대기
                hour_folder = os.path.join(base_folder, current_date, f"{hour:02}")
                if os.path.exists(hour_folder):
                    while not process_videos_in_folder(hour_folder, model, counter, classes_to_count):
                        time.sleep(1)  # 현재 시간 폴더의 모든 비디오를 처리한 후 잠시 대기 (1초)
    
        else:
            print("현재 시간은 5시부터 17시가 아닙니다. 대기 중...") 
            time.sleep(3600)  # 1시간 대기
            continue


def process_video(video_file, model, counter, classes_to_count, retry_delay=10, max_retries=90):
    retries = 0
    global tracked_objects
    global camera_number

    while retries < max_retries:
        try:
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"비디오 파일을 읽을 수 없습니다: {video_file}, 재시도 중... ({retries + 1}/{max_retries})")
                retries += 1
                time.sleep(retry_delay)
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = 1
            frame_skip = 4
            frame_counter = 0

            while cap.isOpened():
                success, im0 = cap.read()
                if not success:
                    print("프레임을 읽는 데 실패했습니다.")
                    break

                try:
                    if frame_counter % frame_skip == 0:
                        results = model.track(im0, persist=True, show=False, classes=classes_to_count)
                        im0 = counter.start_counting(im0, results)
                        boxes = results[0].boxes
                        class_names = results[0].names

                        for box in boxes:
                            cls_id = int(box.cls)
                            obj_type = class_names[cls_id]
                            obj_id = int(box.id) if box.id is not None else None
                            if obj_id is not None:
                                if cls_id in classes_to_count and obj_id not in tracked_objects:
                                    print(f"{obj_type} 감지됨, ID: {obj_id}")
                                    save_capture2(im0, obj_type, model, counter, classes_to_count, results)
                                    tracked_objects.add(obj_id)

                        resized_im0 = cv2.resize(im0, (800, 600))
                        cv2.imshow(f'CCTV_0{camera_number}', resized_im0)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("사용자 종료 요청. 프로그램을 종료합니다.")
                            break

                    frame_counter += 1

                except Exception as e:
                    print(f"프레임 처리 중 오류 발생: {e}")
                    break

            cap.release()
            cv2.destroyAllWindows()
            print("비디오 파일 처리가 완료되었습니다.")
            return

        except Exception as e:
            print(f"비디오 파일 처리 중 오류 발생: {e}")
            retries += 1
            time.sleep(retry_delay)

    if os.path.exists(video_file):
        print(f"{max_retries}회 재시도 실패. 파일 삭제: {video_file}")
        os.remove(video_file)


def process_videos_in_folder(folder_path, model, counter, classes_to_count, check_interval=30):
    global processed_files
    processed_all = True
    processed_files = load_processed_files()

    wait_start_times = {}  # 파일별 대기 시작 시각
    total_wait_start = None  # 전체 대기 시작 시각

    while True:
        video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mp4')]
        video_files.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])

        any_file_processed = False
        current_time = time.time()

        for video_file in video_files:
            abs_path = os.path.abspath(video_file)

            if abs_path in processed_files:
                continue

            # 안정성 확인 (파일 크기 변화 확인)
            if not is_video_file_stable(video_file):
                if abs_path not in wait_start_times:
                    wait_start_times[abs_path] = current_time
                continue

            # 파일이 열리지 않는 경우 (손상 가능성 or 생성 중)
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"[건너뜀] 파일을 열 수 없습니다 (아직 생성 중이거나 손상됨): {video_file}")
                cap.release()
                continue  # 다음 파일로 건너뜀
            cap.release()

            print(f"Processing video file: {video_file}")
            try:
                process_video(video_file, model, counter, classes_to_count)
                record_processed_file(video_file)
                any_file_processed = True
                wait_start_times.pop(abs_path, None)
            except Exception as e:
                print(f"비디오 처리 중 오류 발생: {e}")

        # 처리된 파일이 하나도 없을 때 → 누적 대기
        if not any_file_processed:
            if total_wait_start is None:
                total_wait_start = time.time()

            waited_total_sec = int(time.time() - total_wait_start)
            minutes, seconds = divmod(waited_total_sec, 60)

            print(f"[대기] 처리할 파일이 없습니다. {check_interval}초 후 다시 확인합니다. 누적 대기 시간: {minutes}분 {seconds}초")
            time.sleep(check_interval)
            continue  # 다음 루프로

        total_wait_start = None
        processed_all = False
        break  # 한 번이라도 처리했으면 상위 루프 재호출

    return processed_all
