import threading
import time
import schedule
from datetime import datetime
from ultralytics import YOLO,solutions
import torch
import utils
import gc


def run_camera_task(camera_number, url, base_output_folder):
     # GPU가 사용 가능한지 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("Using GPU")
    else:
        print("Using CPU")

    # YOLO 모델 초기화
    model = YOLO("yolov8n.pt").to(device)

    # 객체 카운터 설정
    line_points = [(0, 0), (0, 0)]
    classes_to_count = [5]  # 2번:car 5번:bus 7번:truck 

    counter = solutions.ObjectCounter(
        view_img=False,
        reg_pts=line_points,
        names=model.names,
        draw_tracks=True,
        line_thickness=2,
    )

    video_save_thread = None


    def start_video_save_thread():
        nonlocal video_save_thread
        utils.stop_flag = False  # 중지 플래그 초기화
        if video_save_thread is None or not video_save_thread.is_alive():
            video_save_thread = threading.Thread(target=utils.save_video_in_chunks, args=(url, base_output_folder, 180, 'mp4v', 1.0, camera_number))
            video_save_thread.start()
            print(f"Video save thread started for camera {camera_number}")


    def stop_video_save_thread():
        print("Stopping video save thread...")  # 로그 추가
        utils.stop_saving_video()
        if video_save_thread is not None and video_save_thread.is_alive():
            video_save_thread.join()
            print(f"Video save thread stopped for camera {camera_number}")


    def job_scheduler():
        schedule.every().day.at("05:00").do(start_video_save_thread) # 05:00에 비디오 저장 시작
        schedule.every().day.at("17:00").do(stop_video_save_thread) # 17:00에 비디오 저장 중지 
        schedule.every(15).minutes.do(gc.collect)  # 15분마다 가비지 컬렉션 실행

        while True:
            schedule.run_pending()
            time.sleep(1)


    # 현재 시간 확인 후, 실행 시간 내에 있다면 비디오 저장 쓰레드 시작
    now = datetime.now().time()
    start_time = datetime.strptime("05:00", "%H:%M").time()
    end_time = datetime.strptime("17:00", "%H:%M").time() 

    # 17시 이후에는 비디오 저장을 시작하지 않도록 설정
    if start_time <= now < end_time:
        start_video_save_thread()

    # 스케줄러를 위한 쓰레드 시작
    scheduler_thread = threading.Thread(target=job_scheduler)
    scheduler_thread.start()

    # YOLO 처리 실행 (메인 스레드에서 실행)
    utils.process_video_files(base_output_folder, model, counter, classes_to_count)

    # 메인 스레드가 종료되지 않도록 유지
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_video_save_thread()
        scheduler_thread.join()