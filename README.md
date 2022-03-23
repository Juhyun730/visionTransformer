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
- 논문 뒷쪽에 보면 2D-위치 임베딩을 추가해보았지만 더 좋은 성능은 아님. (이미지의 패치단위 레벨에 대해 동작하기 때문에) <br/>-> 그래서 1D-위치 임베딩을 사용<br/><br/><br/>

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
  *Fine-Tuning : 1-linear Layer로 수행<br/><br/><br/>


### 3. METHOD: Transformer 구조<br/>
![image](https://user-images.githubusercontent.com/52825569/159692681-42e13390-d77d-492b-9986-04d32c8f53d3.png)<br/>
* 위의 그림은 VIT에서 쓰인 트랜스포머 구조<br/>
![image](https://user-images.githubusercontent.com/52825569/159693050-9bfc9bdf-5510-435e-bf40-d557e36f3b60.png)<br/>
* 위의 그림은 트랜스포머 구조를 수식으로 나타낸 것<br/><br/>

#### Transformer Encoder<br/>
- 모든 Block 이전에 LayerNorm(LN) 적용<br/>
    * LN은 배치에 있는 각각의 input의 feature들에 대한 “평균과 분산”을 구해서 정규화하는 것<br/>
- Residual Connection 모든 블록 끝에 적용<br/>
   * Residual : 한 블록이 끝날 때, input x를 더하는 과정<br/>
- 첫번째 Block: Multi-Head로 구성된 self-Attention 메커니즘이 적용됨(2번 수식)
- 두번째 Block: MLP BLOCK(다중 퍼셉트론)  (3번 수식)
- GELU((Gaussian Error Linear Unit)라는 활성화 함수를 사용(mlp는 linear layer-> gelu->dropout->linear layer 구조)<br/>


### 4. METHOD: Inductive bias & Hybrid Architecture
#### Inductive bias
- inductive bias는 학습 시에는 만나보지 않았던 상황에 대해 정확한 예측을 하기 위해 사용하는 추가적인 가정을 의미.<br/>
   즉, 모델이 목표함수를 학습하고 훈련 데이터를 넘어 다른 데이터에 대해서도 일반화할 수 있는 능력을 가질 수 있게끔 만드는 상황
- CNN, RNN에서는 지역성(locality)이 전체 모델에 걸쳐 각 레이어로 적용된다. 
*CNN의 경우 filter를 통해 특징을 뽑아내면 local적인 부분에서만 특징을 뽑아낼 수 있다. RNN의 경우에서는 문장의 시퀀스때문에 가까운 애들한테 더 많은 영향을 준다.<br/> => global한 영역의 처리가 어려움
- VIT에서는 MLP계층만 local적이고, 다른 계층은 모두 global(전역적)하다.<br/>
*positional embedding(위치임베딩)을 통해 모든 정보를 활용함<br/> 그래서 VIT는 CNN과 다르게 inductive bias가 훨씬 적다.

<span style="color: red"> => Cnn처럼 locality에 대한 가정은 없기때문에, VIT는 더 많은 양의 데이터를 통해, 원초적인 관계를 ROBUST하게 학습시켜야 함. (어떤 상황에서도 잘 예측하게끔)</span>

#### Hybrid Architecture
- 이미지 패치를 그대로 사용하는 대신, CNN의 feature map으로부터 sequence를 뽑아내 사용하는 것.
과정: Cnn으로 feature를 추출하고 <br/>
-> flatten시킨 다음 
<br/>-> 패치 임배딩(입력임베딩에서의 linear projection)의 투영법을 통해 패치를 뽑아냄.


### 그 외 기법들
#### 높은 해상도로 Fine-Tuning
- 사전 훈련 때보다 더 높은 해상도로 Fine-Tuning 하는 것이 성능에 도움이 됨
- 더 높은 해상도와 Patch Size가 동일하면, sequence의 길이가 증가하게 된다. 이렇게 되면, fine tuning 시 pre-trained position embedding은 더 이상 의미가 없어지게 된다..
 <br/>=>그래서 2d interpolation(2차원 보간법)을 사용한다. <br/><br/><br/>

## EXPERIMENTS
실험에서는 resnet, vit, 하이브리드 세가지 모델의 학습능력을 평가하였다.<br/>
데이터는 다음과 같다.<br/>
### 데이터 
사전 학습데이터셋: <br/>
1. ImageNet-1k클래스(1.3 M)<br/>   
2. ImageNet-21k클래스(14 M)<br/>   
3. JFT-18k클래스(303M) *1k는 1000개 클래스를 의미, 1M는 100만장을 의미<br/>
*Fine tuning 데이터 셋: ImageNet, CIFAR-10/100, Oxford-IIT Pets<br/>
*Transfer 데이터 셋: VTAB(Visual Task Adaptation Benchmark) (구글에서 발표한 적은 데이터셋을 활용한 Transfer Learning 성능 평가를 위한 데이터셋) <br/>
 VTAB는 다음과 같다.<br/>
 ** - Natural : Pet, CIFAR<br/>
 ** - Specialized : Medical이나 위성 이미지<br/>
 ** - Structured : 기하학(Geometric)과 지역적(Localization)인 이해가 필요

### 실험에 사용된 VIT 모델
![image](https://user-images.githubusercontent.com/52825569/159695465-e1894e66-2729-43e4-80a4-2e564d8612f6.png)<br/><br/>
vit는 BERT에 사용된 구성을 기본으로 실험: 모델은 총 세 개를 이용하였다!<br/>
*여기서 ViT-L/16 는 "Large" 매개변수 크기 및 16x16 Input Patch 사용한 것이다.<br/><br/>

옵티마이저, 배치사이즈, 이미지사이즈는 다음과 같다.<br/>
![image](https://user-images.githubusercontent.com/52825569/159695629-59c2b69b-ee99-49d2-8752-b90499c64c9e.png)<br/><br/><br/>

#### 1번째 실험: SOTA와의 비교
![image](https://user-images.githubusercontent.com/52825569/159695831-25eb3937-6e5a-4cab-af20-71379c9f86b0.png)<br/>
- BIT(bit transfer), NOISY STUDENT는 기존의 다른 CNN SOTA모델
- 실험은 모델을 TPUv3라는 하드웨어로 JFT-300M로 사전학습 시킨 경우의 다른 SOTA 모델들과의 정확도 비교한 것
- 실험 결과, vit-H/14가 제일 성능이 좋은 것을 알 수 있음
- 계산 시간도 또한 vit-H/14는 BIT와 NoisyStudent보다 4~5분의 1, vit-L/16은 약 15분의 1정도 절약된 것을 알 수 있음<br/><br/>

![image](https://user-images.githubusercontent.com/52825569/159696075-453e2926-6520-4faf-8d61-f9cf2d590177.png)<br/>
데이터셋 설명<br/>
 *Natural : Pet, CIFAR<br/>
 *Specialized : Medical이나 위성 이미지<br/>
 *Structured : 기하학(Geometric)과 지역적(Localization)인 이해가 필요<br/>
- VIVI와 S4L(Self-supervised semi-supervised learning)는 이전의 다른 CNN SOTA모델.
- 위의 4가지 모델 모두  ImageNet으로 사전 학습된 모델임
- 실험은 모델들을 VTAB 태스크에 대하여 Natural, specialized, structured분야로 전이학습 한 결과
- Specialized 태스크에서 VIT와 BIT가 비슷할 뿐, 나머지 태스크들에서는 VIT가 압도적인 성능을 보임<br/><br/>

#### 2번째 실험: 데이터 사이즈, 계산 비용
![image](https://user-images.githubusercontent.com/52825569/159696733-6cc23081-e307-4445-8099-2b66392a4266.png)
- 사전학습을 ImageNet-1k(1.3 M) / ImageNet-21k(14 M) / JFT-18k(303M)로 진행, Fine tunning은 imagenet-1k로 진행
- 작은 데이터셋에서의 최적화를 위해서, 정규화 설정 : Weight Decay, Dropout, Label Smoothing 기법 적용
- 결과는 사전학습 시 데이터 셋이 클수록 ViT의 성능이 좋고, 적은 데이터 셋에서는 좋지 않음<br/><br/>

![image](https://user-images.githubusercontent.com/52825569/159696815-75a4f2ba-92bf-46ec-be11-68d96761ab91.png)
- 사전학습을 JFT-18k 데이터 셋의 9M, 30M, 90M의 하위 데이터집합을 가지고 진행
- 모델의 본질적인 성능을 평가하기 위해 정규화는 안함
- RESNET은 적은 데이터셋에서는 성능이 좋지만, 큰 데이터셋에서는 VIT가 더 좋은 성능을 가짐
- 적은 데이터에서 동등한 계산 비용으로, vit가 resnet보다 더 과적합이 잘됨.<br/><br/>

#### 3번째 실험: hybrid와의 비용대비 효과 비교
Resnet(BIT)과 VIT, Hybrid모델을 JFT-300M으로 사전학습 시킨 다음, 이 때의 계산량과 전이학습에서의 정확도를 비교한 실험<br/>
![image](https://user-images.githubusercontent.com/52825569/159697086-ad28447d-d767-4fad-9738-5a8d391f8162.png)
- ViT은 BiT보다 비용대비 효과가 좋다(저비용으로도 같은 성능을 달성한다). 
- 비용이 한정되어 있는 경우 Hybrid가 제일 효과적이다.
- 논문 저자는 VIT의 성능이 아직 포화되지 않아서 더 업그레이드된 성능향상을 기대할 수 있다고 말한다.<br/><br/><br/><br/>


## INSPECTING VISION TRANSFORMER

![image](https://user-images.githubusercontent.com/52825569/159697368-0955add9-b635-471f-ac10-041758f2c956.png)
끝에 논문에서는 다음과 같은 시각화 제료를 첨부한다.<br/>
- 첫번째 사진: VIT-L/32모델의 학습된 linear embedded 필터 중 맨 처음에나오는 28개 필터를 시각화 한 것<br/>
- 두번째 사진: VIT-L/32모델에서 각각의 패치의 위치 임베딩의 행과 열에 따른 코사인 유사도를 구한 것<br/>
*옆의 그림은 위치 (1,1)의 위치 임베딩과 다른 위치 임베딩들의 유사도를 전부 표시한 것이다.<br/><br/>

![image](https://user-images.githubusercontent.com/52825569/159697833-9a151caf-b9f7-4b7a-95fb-193eb78548a0.png)
위의 사진은 어텐션 distance를 시각화한 표이다. <br/>
*Attention distance란?<br/>
attention 가중치에 의해 통합되는 이미지 공간내의 평균거리를 계산한 것.<br/>
Cnn에서의 receptive field와 같은 것이라고 생각하면 됨.(어텐션이 적용되는 영역의 크기)<br/>

표를 나름대로 해석을 해보자면 다음과 같다.(아닐수도 있다.)
- 네트워크 층이 적을 때는 몇몇 어텐션 헤드들이 작은 어텐션 거리를 갖는다.<br/>
=> 작은 어텐션 거리를 갖는 다는 것은 local적인 부분을 어텐션한다는 것을 의미(cnn의 필터와 같이 local적으로 어텐션함)<br/>

- 층이 깊어질수록 어텐션 거리가 커지게 된다.<br/>
=> vit가 이미지를 attention할 때, 더 global적으로 반영해서 attention을 하게 됨을 의미<br/><br/><br/>

## SELF-SUPERVISION & CONCLUSION
이 논문에선 마지막으로 self-supervision 에 대해서 조금 설명하고있다.<br/>
### SELF SUPERVISED LEARNING
- 트랜스포머 기반 연구들은 self-supervised pre-training에서도 많은 성공을 보인다.<br/>
- 이 논문엔서는 BERT에서 사용된 masked language modeling task를 모방하여 ‘masked patch prediction for self-supervision’를 실험하였고, 그 결과 ViT-B/16 모델은 image-net 데이터에서 79.9 % 정확도를 보일 수 있었다고 말한다.<br/>
- 이 때 masked patch 실험과정은 다음과 같다.<br/>
*1.이미지 패치 중 50%를 mask로 가리거나 아예 다른 패치로 대체한다. <br/>
*2. 평균 3bit색상을 예측, 4 × 4 downsized된 패치 예측(16x16패치기준), 패치의 Regression 예측을 진행함.<br/>

Supervised learning 방식보다는 정확도가 낮지만, 이 분야에서 앞으로의 연구가 기대된다고 말한다.<br/><br/>


### CONCLUSION
- 비전분야에 self-attention을 사용하는 이전의 연구들과는 다르게, 본 연구에서는 모델을 설계할 때 어떠한 inductive bias도 추가하지 않았다는 의의를 가지고 있다.
- Large Dataset(JFT-300M 급)에서 사전학습을 시킬 경우, 높은 정확도를 보인다.
- 사전학습 시, 계산 비용이 상대적으로 저렴하다.
- 이미지 Detection, segmentation과 self- supervised learning 분야에서의 연구가 기대되며, 모델을 다양하게 scaling할 경우 성능향상 또한 기대된다.


