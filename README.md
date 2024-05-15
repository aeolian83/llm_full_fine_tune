# LLM_practice

-   This repository is for practicing and recording various LLM codes related to training, inference, and more for LLMs.
-   This repository also aims for full fine-tuning (unquantized fine-tuning) of the LLM, with plans to methodically proceed from data preparation, preprocessing, model training, to model uploading.
-   PyTorch will be used as the primary library, and the use of other libraries for distributed training will be indicated separately.

## 00 QLoRa

### 1) LLAMA2-KO Instruction fine tune

-   준범님의 llama2-ko-7b 모델을 인스럭션 파인튠하기 위한 코드
-   QLoRA훈련을 위한 [코드](./00_QLoRa_fine_tune/01_QLoRA_08.ipynb)와 LoRA merge [코드](./00_QLoRa_fine_tune/02_QLoRA_Merge_upload_08.ipynb) 및 테스트 인퍼 [코드](./00_QLoRa_fine_tune/04_test_inference05.ipynb)로 되어 있음

#### a. 이슈

1. 처음 시도한 몇몇의 훈련에서 eos_token을 생성해내지 못하고, 같은 말을 반복하는 문제 발생
   해결: 올바른 훈련코드와 데이터, 그리고 pad 토큰을 추가하고 모델의 임베딩 레이어를 리사이즈 한뒤에 훈련 하는 것으로 해결

## 01 Chat Vector

### 1) Chat vector Inference

-   chat-vector 논문[[link](https://arxiv.org/abs/2310.04799v2)]에 의하면 모델 구조가 같다면 parameter의 different를 다른 모델에 더함으로써 다른언어로 cp train한 모델의 differnt를 적용해서 원어로 instruction된 모델을 다른언어로 instruction한 모델로 만들 수 있다고 주장하였다.
-   이 방법론에 대한 실험 부분

#### a. CP(Continual Pre-trained) transfer

-   chat-vector는 instruction 모델의 매개변수를 cp모델로 전이해서 cp모델의 instruction모델을 만들 수 있다고 주장하였다.
-   그렇다면 다른 방법으로 instruction 모델에 cp 모델의 매개변수를 전이해서 영어로 만든 instruction 모델을 다른언어 instruction모델로 전이 가능 할 것이라 생각하였다.

#### b. 이슈

-

#### b. 레퍼런스

1. https://github.com/ashishpatel26/LLM-Finetuning/blob/main/7.FineTune_LLAMA2_with_QLORA.ipynb
2. https://huggingface.co/blog/4bit-transformers-bitsandbytes
3. https://pytorch.org/blog/finetune-llms/
4. https://colab.research.google.com/drive/1vIjBtePIZwUaHWfjfNHzBjwuXOyU_ugD?usp=sharing#scrollTo=6k_nL6xJMZW2
