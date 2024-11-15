# TODO : VSCode 에디터에서 파이썬 환경 설치 (2024.11.12 jbh)
# 참고 URL - https://icedhotchoco.tistory.com/entry/VS-Code-Install


# TODO : VSCode 에디터에서 패키지 numpy 설치 
# 주의사항 - tensorflow와 호환성 문제로 numpy의 버전은 최신 버전이 아니라 버전 1.26.0 으로 설치
# 참고 URL - https://numpy.org/news/#numpy-1260-released
# 참고 2 URL - https://rigun.tistory.com/90
# 참고 3 URL - https://wikidocs.net/227412

# TODO : VSCode 에디터에서 패키지 numpy 버전 1.26.0 설치시 오류 메시지 "ERROR: Could not install packages due to an OSError: [WinError 2] 지정된 파일을 찾을 수 없습니다: 'C:\\Python312\\Scripts\\f2py.exe' -> 'C:\\Python312\\Scripts\\f2py.exe.deleteme'"
#        출력 원인 파악 및 해당 오류 해결하기 위해 터미널 명령어 "pip install numpy==1.26.0 --user" 를 사용하여 특별한 권한 없이 패키지 numpy 버전 1.26.0를 설치 진행 (2024.11.12 jbh)
# 참고 URL - https://stackoverflow.com/questions/66322049/could-not-install-packages-due-to-an-oserror-winerror-2-no-such-file-or-direc

# TODO : VSCode 에디터에서 패키지 tensorflow 설치시 오류 메시지 "ERROR: Could not install packages due to an OSError: [WinError 2] 지정된 파일을 찾을 수 없습니다: 'C:\\Python312\\Scripts\\wheel.exe' -> 'C:\\Python312\\Scripts\\wheel.exe.deleteme'"
#        출력 원인 파악 및 해당 오류 해결하기 위해 터미널 명령어 "pip install tensorflow --user" 를 사용하여 특별한 권한 없이 패키지 tensorflow 설치 진행 (2024.11.12 jbh)

# TODO : VSCode 에디터에서 패키지 tensorflow 설치 후 패키지 import 처리 오류 메시지 "import tensorflow as tf 지정된 모듈을 찾을 수 없습니다" 해결하기 위해 
#        VSCode 에디터 좌측 하단 톱니바퀴 모양 버튼 클릭 -> 컨텍스트 메뉴 "설정" 클릭
#        -> 화면 "설정" 출력 -> 검색어 "python" 입력 및 엔터 -> 화면 "설정" 하위 항목 "사용자" 클릭 -> 하위 항목 "확장" 클릭 -> 하위 항목 "Python" 클릭 
#        -> 화면 "설정" 중앙에 출력된 설정들 중 설정 "Python › Terminal: Activate Environment"에 체크박스에 체크된 사항 해제 및 저장하여 오류 해결 (2024.11.12 jbh)
# 참고 URL - https://webnautes.tistory.com/2032
# 참고 2 URL - https://qna.programmers.co.kr/questions/5393/tensorflow-%EB%AA%A8%EB%93%88%EC%9D%84-%EB%B6%88%EB%9F%AC%EC%98%A4%EC%A7%80%EB%A5%BC-%EB%AA%BB%ED%95%98%EB%8A%94-%EC%98%A4%EB%A5%98
# 참고 3 URL - https://code.visualstudio.com/docs/python/environments

import numpy as np 
import tensorflow as tf 
import keras
print(np.__version__)
print(tf.__version__)
print(keras.__version__)
print('Hello Python!')
