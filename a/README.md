# 얼굴 검출 예제

OpenCV의 Haar cascade로 정면 얼굴을 찾아 초록색 사각형과 얼굴 개수를 표시합니다.
개인의 신원을 구분하는 기능은 포함하지 않습니다. 옆얼굴, 작은 얼굴, 어두운 환경에서는 검출이 누락될 수 있습니다.

저장소 루트에서 다음 명령을 실행하세요.

```powershell
python -m pip install -r a/requirements.txt

# 기본 웹캠 (Q, Esc 또는 창 닫기로 종료)
python a/faceDetection.py

# 다른 카메라
python a/faceDetection.py --camera 1

# 이미지 파일 (아무 키로 창 닫기)
python a/faceDetection.py --image "photo.jpg"

# 창 없이 검출 결과 저장 (출력 폴더는 미리 생성)
python a/faceDetection.py --image "photo.jpg" --output "result.jpg" --no-display
```

카메라를 열 수 없다면 다른 프로그램의 카메라 사용 여부와 Windows 카메라 접근 권한을 확인하세요.
이미지 경로는 명령을 실행한 폴더 기준이며, 절대 경로도 사용할 수 있습니다.
