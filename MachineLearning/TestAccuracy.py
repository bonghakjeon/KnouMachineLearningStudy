
# TODO : 2가지 패키지 "keras.src", "keras.api" 차이점이 무엇인지 확인하기 (2024.11.14 jbh)
from keras.src.models.sequential import Sequential
from keras.src.layers import Dense
from keras.src.datasets import fashion_mnist

from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.datasets import fashion_mnist
