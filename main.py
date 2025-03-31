import multiprocessing
import time
import psutil
from YOLOv8_Counting import run_camera_task  # YOLOv8_Counting에서 함수 임포트


def set_cpu_affinity(process, core_ids):
    p = psutil.Process(process.pid)
    p.cpu_affinity(core_ids)
    print(f"프로세스 {process.pid}가 CPU 코어 {core_ids}에 할당되었습니다.")


def start_camera_process(camera_number, url, base_output_folder, core_ids):
    process = multiprocessing.Process(target=run_camera_task, args=(camera_number, url, base_output_folder))
    process.start()
    #set_cpu_affinity(process, core_ids)
    return process


if __name__ == "__main__":
    # 카메라 1 설정
    camera_1_number = 1
    url_1 = 'rtsp://admin:joa0102!!@192.168.0.35:553'
    base_output_folder_1 = "saved_videos_01"
    core_ids_1 = [0]  # 프로세스 1에 할당할 CPU 코어 (두 개)

    # 카메라 2 설정
    camera_2_number = 2
    url_2 = 'rtsp://admin:joa0102!!@192.168.0.37:554'
    base_output_folder_2 = "saved_videos_02"
    core_ids_2 = [2]  # 프로세스 2에 할당할 CPU 코어 (두 개)

    # 각각의 카메라 프로세스를 시작하고 CPU 코어에 할당합니다.
    process1 = start_camera_process(camera_1_number, url_1, base_output_folder_1, core_ids_1)
    process2 = start_camera_process(camera_2_number, url_2, base_output_folder_2, core_ids_2)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, terminating processes...")
        process1.terminate()
        process2.terminate()

        process1.join()
        process2.join()