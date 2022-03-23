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
![image](https://user-images.githubusercontent.com/52825569/159668606-e3d456d0-6508-439c-98a7-6f6d1e74f5f1.png)<br/><br/>

### 1. METHOD: 트랜스포머에 맞게끔 데이터 변경!<br/>  
수식은 아래와 같다. <br/>     
![image](https://user-images.githubusercontent.com/52825569/159668913-97c09ba3-6f19-4821-9989-14d62053fea8.png)<br/><br/>    

#### 트랜스포머에 맞게끔 데이터 변경!: 패치 임베딩<br/>   
![image](https://user-images.githubusercontent.com/52825569/159690353-5daf279d-3be4-4b5b-ba7a-7ad8745ef655.png)  <br/>


- 2차원의 이미지는 이미지(H*W*C) 구조를 N개의 패치 형태로(N*(P^2*C))로 변형시킨다.  
   (P, P)는 이미지 패치의 해상도(h,w)이고, N=(H*W)/P^2 이므로    이미지 패치의 개수이다.  
 
- 오리지널 트랜스포머는 input으로 1D의 토큰 임베딩 시퀀스를 가지기 때문에, 2D 이미지를 다루기 위해 패치를 자른 것을 flatten하고, linear projection을 사용하여 1차원에 mapping한다. <br/>-> 이 결과값을 patch embedding이라 부른다. <br/>

- 아래에 더 자세한 그림을 첨부하였다.<br/>
 
 ![image](https://user-images.githubusercontent.com/52825569/159691591-d5111121-cde3-4ea3-8205-320304e1a60c.png)<br/><br/>
 
 #### 트랜스포머에 맞게끔 데이터 변경!: 위치 임베딩: Epos
- 위치 임베딩은 위치정보를 유지하기 위해 패치 임베딩에 더해준 값
- 논문 뒷쪽에 보면 2D-위치 임베딩을 추가해보았지만 더 좋은 성능은 아님. (이미지의 패치단위 레벨에 대해 동작하기 때문에) <br/>-> 그래서 1D-위치 임베딩을 사용

### 2. METHOD: Cls 토큰과 classification head<br/>
![image](https://user-images.githubusercontent.com/52825569/159691934-970f5abb-a16c-4d24-bf57-7cf10fb717ff.png)<br/>
#### CLS 토큰
- CLS토큰은 BERT의 Class 토큰처럼 학습 가능한 Embedding patch ( z0=xclass )를 추가로 붙여 넣은 것
    <br/> *BERT에서는 CLS(classificataion) token을 두어 이 token에 대한 output을 분류에 활용한다.

- 이 패치에 대해 나온 인코더 아웃풋은 이미지 representation으로 해석하여 분류에 사용한다. 즉, Classifcation head에 Cls 토큰을 넣음으로써 이미지에 대한 예측이 나온다.


#### Classification Head<br/>
![image](https://user-images.githubusercontent.com/52825569/159692216-5fff3c86-b36f-4b6a-9799-c99b7dbe5e84.png)

- 이미지의 예측을 출력하는 부분<br/>
  *Pre-training : 1-hidden Layer인 MLP(다중퍼셉트론 층)으로 수행<br/>
  *Fine-Tuning : 1-linear Layer로 수행<br/>




