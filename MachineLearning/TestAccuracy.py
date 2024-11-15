import keras

# TODO : 2가지 패키지 "keras.src", "keras.api" 차이점이 무엇인지 확인하기 (2024.11.14 jbh)
from keras.src.models.sequential import Sequential
from keras.src.layers import Dense
from keras.src.datasets import fashion_mnist

# from keras.api.models import Sequential
# from keras.api.layers import Dense
# from keras.api.datasets import fashion_mnist

import numpy as np 

# TODO : 파이썬 소스 파일 "TestAccuracy.py" 실행시 오류 메시지 "ModuleNotFoundError: No module named 'matplotlib'" 출력됨
#        해당 오류 해결 하기 위해서 터미널 명령어 "pip install matplotlib --user" 사용해서 파이썬 추가 패키지 'matplotlib' 설치함 (2024.11.15 jbh)
# 참고 URL - https://tech.zinnunkebi.com/programming-language/python/python-modulenotfounderror-matplotlib/
import matplotlib.pyplot as plt 

# 2. 데이터 시각화에 쓰일 변수 
A = 111
B = 10
C = 25
D = 5

# 3. 데이터 스케일링에 쓰일 변수 
E = 255.0

# 4. 모델 구성에 쓰일 변수 
DENSE_A = 28
DENSE_B = 128
DENSE_C = 64
DENSE_D = 10

# 5. 모델 컴파일 및 학습에 쓰일 변수 
FIT_A  = 10
FIT_B = 64 
FIT_C = 1

# 6. 모델 테스트 
PRE_A = 0

# 1 단계 - 데이터 셋 import - Fashion MNIST 
# 참고 URL - https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # fashion_mnist 데이터 셋 로드

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] # 패션 카테고리 이름 리스트 객체 class_names 선언 및 값 할당 

# 2 단계 - 데이터 시각화
# 함수 plt.figure() - 새 그림을 만들거나 기존 그림 활성화
# 참고 URL - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
plt.figure()   

# 함수 plt.imshow - 데이터를 2D 래스터 이미지로 표시
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
    ----- plt.subplot(5,5,i+1)
    ----- plt.xticks([])
    ----- plt.yticks([])
    ----- plt.imshow(train_images[i], cmap=plt.cm.binary)
    ----- plt.xlabel(class_names[train_labels[i]])

plt.show() # 데이터 시각화 하기 위해 화면 출력 

# 3 단계 - 데이터 스케일링
train_images = train_images / 255.0
test_images = test_images / 255.0

# 4 단계 - 모델 구성
model = Sequential()
model.add(keras.Input(shape=(28, 28)))       # 입력 데이터의 형태 지정
model.add(keras.layers.Reshape((28 * 28,)))  # 입력 데이터를 1차원 벡터로 변환
model.add(Dense(128, activation='relu'))     # 은닉층 추가(노드 수 128개, ReLU 활성화 함수)
model.add(Dense(64, activation='relu'))      # 은닉층 추가(노드 수 64개, ReLU 활성화 함수)
model.add(Dense(10, activation='softmax'))   # 출력층 추가(10개의 클래스 분류, 활성화 함수 softmax)
model.summary()

# 5 단계 - 모델 컴파일 + 학습
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, batch_size=64, verbose=1)

# 6 단계 - 모델 테스트
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))