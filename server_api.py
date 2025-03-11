import requests


def file_upload( unitid, file_path ):
 
    # 업로드할 파일을 읽기 모드로 열기
    files = {'file': open(file_path, 'rb')}

    # API URL 설정
    url = "https://api.joa-iot.com/cms/unit/upload"

    # POST 요청으로 파일 업로드
    data = {   'unitid': unitid,
	           'type': 'cctv',
               'description': 'CCTV footage upload' }
    response = requests.post(url, files=files, data=data)

    # 결과 출력
    if response.status_code == 200:
        print('업로드 성공:', response.json())
    else:
        print('업로드 실패:', response.status_code, response.text)

    # 파일 닫기
    files['file'].close()


