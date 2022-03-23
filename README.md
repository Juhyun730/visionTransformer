# visionTransformer


비전 분야에 트랜스포머 구조를 처음 적용한 논문


## Introduction
- 그동안 large dataset으로 훈련된 트랜스포머의 pre-trained모델은 NLP분야에서 많이 채택되고 있다.
- 컴퓨터 비전 분야에서 NLP의 영감을 받아 CNN과 self-attention이 결합이 된 연구들이 많이 진행되어왔지만, 현대의 하드웨어 가속기에서는 비효율적이다.
- Large scale 이미지 인식분야에서는 아직까지는 Resnet 구조가 SOTA이다.

그래서 이 논문에서는 성공적인 성능의 트랜스포머의 영감을 받아, 트랜스포머를 있는 그대로 가져와서 이미지에 적용을 하게 된다.

-> 이미지를 잘라서 patch(트랜스포머의 word와 같은 역할)로 만들고 sequence를 linear embedding으로 만들어 실험하여서 좋은 결과를 달성하는 것이 이 실험의 목적!


## Related Work
- 기존에는 대표적으로 BERT, GPT라는 트랜스포머 모델들이 있다.
- Cordonnier라는 사람은 2x2 이미지 패치를 추출하여 self-attention을 사용했다. 하지만 이미지 패치의 사이즈가 작아서 작은 해상도에만 적용 가능하다는 단점이 있다.
- 본 연구와 가장 비슷한 이전 모델인 ‘Igpt’라는 모델은 이미지의 해상도와 color space를 줄인 뒤 image 픽셀들에 트랜스포머를 적용한 모델이고, 하나의 generative model로써 비지도 방식으로 훈련되었으며, 정확도는 약 72%를 기록한다.
- 아직까지는 전체 크기의 이미지에 대해 global self-attention을 수행하는 트랜스포머의 사례는 없다.
등등의 트랜스포머 모델들과 cnn에 관한 기존 연구 소개

-> 이 논문은 위에서 나온 어텐션에 이미지를 적용시킨 다른 연구들보다 더 성능이 좋다는 실험결과를 얻음


## METHOD
![image](https://user-images.githubusercontent.com/52825569/159668537-d6ac736e-f646-4aad-b742-d38a7ebd66ab.png)

구조는 다음과 같은 오리지널 트랜스포머의 인코더 구조를 그대로 갖다가 사용한다. 
![image](https://user-images.githubusercontent.com/52825569/159668606-e3d456d0-6508-439c-98a7-6f6d1e74f5f1.png)

### METHOD: 트랜스포머에 맞게끔 데이터 변경!
![image](https://user-images.githubusercontent.com/52825569/159668913-97c09ba3-6f19-4821-9989-14d62053fea8.png)
