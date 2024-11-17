# TODO : 방송대 머신러닝 출석수업 소스코드 구현 (2024.11.18 jbh)
# GitHub - Repositories
# 참고 URL - https://github.com/bonghakjeon/KnouMachineLearningStudy
import keras

# TODO : keras.src와 keras.api의 차이 확인 및 keras.api 사용 하도록 구현 (2024.11.18 jbh)
# 참고 URL - https://chatgpt.com/c/673a6d16-e878-8011-9796-de1309d52fd7
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.datasets import fashion_mnist
# from keras.src.models.sequential import Sequential
# from keras.src.layers import Dense
# from keras.src.datasets import fashion_mnist

import numpy as np 

# TODO : 파이썬 소스 파일 "TestAccuracy.py" 실행시 오류 메시지 "ModuleNotFoundError: No module named 'matplotlib'" 출력됨
#        해당 오류 해결 하기 위해서 터미널 명령어 "pip install matplotlib --user" 사용해서 파이썬 추가 패키지 'matplotlib' 설치함 (2024.11.15 jbh)
# 참고 URL - https://tech.zinnunkebi.com/programming-language/python/python-modulenotfounderror-matplotlib/
import matplotlib.pyplot as plt 

print(np.__version__)      # numpy 버전 1.26.0
print(keras.__version__)   # keras 버전 3.6.0

# 1 단계 - 데이터 셋 import - Fashion MNIST 
# 참고 URL - https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # fashion_mnist 데이터 셋 로드

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] # 패션 카테고리 이름 리스트 객체 class_names 선언 및 값 할당 

# 2 단계 - 데이터 시각화
# 함수 plt.figure() - 새 그림을 만들거나 기존 그림 활성화
# 참고 URL - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
plt.figure()   

# 함수 plt.imshow - 데이터를 2D 래스터 이미지로 표시(원하는 사이즈의 픽셀을 원하는 색으로 채워서 그림 만들기)
# 참고 URL - https://pyvisuall.tistory.com/78
plt.imshow(train_images[111]) 

# 함수 plt.colorbar() - 그래프 옆에 Color Bar 추가 
# 참고 URL - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
plt.colorbar()

# 함수 plt.grid(True) - Grid 그리기 
# 참고 URL - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.grid.html
plt.grid(True) 

# 함수 plt.figure() - 새 그림을 만들거나 기존 그림 활성화
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)   # 현재 그림에 축을 추가하거나 기존 축 검색 / 참고 URL - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html
    plt.xticks([])   # xticks 비활성화 / 참고 URL - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xticks.html
    plt.yticks([])   # # yticks 비활성화 / 참고 URL - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.yticks.html
    plt.imshow(train_images[i], cmap=plt.cm.binary)   # 함수 plt.imshow - 데이터를 2D 래스터 이미지로 표시(원하는 사이즈의 픽셀을 원하는 색으로 채워서 그림 만들기)
    plt.xlabel(class_names[train_labels[i]]) # X축의 label 설정 / 참고 URL - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xlabel.html

# TODO : 아래 주석친 코드 "plt.show()" 주석 해제해야 화면 2개(Figure 1, Figure 2)가 출력되고 해당 화면 2개 종료(닫기 "X" 버튼 클릭)시 
#        아래 터미널 창에서 모델 컴파일/학습/테스트 결과가 출력 된다. 
#        만약 주석 처리를 하게 되면 화면 2개(Figure 1, Figure 2)는 출력 되지 않고 아래 터미널 창에서 모델 컴파일/학습/테스트 결과가 출력 된다. 
plt.show() # 데이터 시각화 하기 위해 화면 출력 

# 3 단계 - 데이터 스케일링
train_images = train_images / 255.0
test_images = test_images / 255.0

 # 4 단계 - 모델 구성
# 참고 URL - https://wikidocs.net/192931
# 모델을 순차적으로 쌓을 수 있는 Sequential 클래스 데이터 모델 객체 seqModel 생성
# 참고 URL - https://www.tensorflow.org/guide/keras/sequential_model?hl=ko
seqModel = Sequential() 
seqModel.add(keras.Input(shape=(28, 28)))      # 입력 데이터 형태 지정 (28x28 픽셀 2차원 배열 형태)
seqModel.add(keras.layers.Reshape((28 * 28,)))  # 입력 데이터(28x28 픽셀 2차원 배열 형태) -> 1차원 벡터(784,) 변환 (Dense Layer에서 1차원 백터 입력 데이터가 필요하기 때문에 변환 필요)
seqModel.add(Dense(128, activation='relu'))     # 첫 번째 은닉층 추가(노드 수 128개, ReLU 활성화 함수 사용) 
seqModel.add(Dense(64, activation='relu'))      # 두 번째 은닉층 추가(노드 수 64개, ReLU 활성화 함수 사용) - 첫 번째 은닉층에서 추출된 특성을 가지고 추가적인 패턴 학습 진행
seqModel.add(Dense(10, activation='softmax'))   # 출력층 추가(10개의 클래스(노드) 분류, 활성화 함수 softmax 사용)   
seqModel.summary()  # 모델의 구조를 간략하게 요약하여 보기(출력)


# 5 단계 - 모델 컴파일 + 학습
# 참고 URL - https://blog.naver.com/handuelly/221822938182
# 모델 구성 완료 후 compile() 메서드를 호출해서 모델 학습 과정 설정(모델을 빌드하고 실행하기 전에 컴파일 하는 훈련 준비 단계)
seqModel.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
# 모델 학습 과정 설정 완료 후 fit() 메서드 호출해서 주어진 epoch 수 만큼 모델 학습 진행 (epoch 수 만큼 전체 입력 데이터 순회)
seqModel.fit(train_images, train_labels, epochs=10, batch_size=64, verbose=1)

# 6 단계 - 모델 테스트
# 참고 URL - https://velog.io/@ksolar03/%EB%94%A5%EB%9F%AC%EB%8B%9D-tf.keras-%EC%A3%BC%EC%9A%94-%EB%AA%A8%EB%93%88-%EC%A0%95%EB%A6%AC
test_loss, test_acc = seqModel.evaluate(test_images, test_labels)  # 성능 확인 
print('Test accuracy:', test_acc)
predictions = seqModel.predict(test_images) # 예측 
print(predictions[0])
# 함수 np.argmax - 가장 높은 값의 인덱스 찾기 / 참고 URL - https://numpy.org/doc/stable/reference/generated/numpy.argmax.html / 참고 2 URL - https://powerdeng.tistory.com/135
print(np.argmax(predictions[0])) 