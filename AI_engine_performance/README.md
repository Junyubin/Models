# AI_engine_performance
    
    KISA 검증 
    
    ※ HOW_TO_APPLY_ESOINN_to_DTI.docx 참조

    1	개발 환경 설정
        1.1	설치 라이브러리 정보 (for ESOINN)
            ※	dti_v3_essoin.py 이외의 모듈에 대한 필요 라이브러리는 HOW_TO_APPLY_ESOINN_to_DTI.docx 참조
            Process 	Command 	비고 
            1)	conda 환경 설정	conda create -n dti_legacy python=3.7.11	
            2)	설정된 conda 환경 입력	conda activate dti_legacy	
            3)	Scikit-learn 설치	conda install -c anaconda scikit-learn=0.22.1	
            4)	Matplotlib 설치	conda install -c conda-forge matplotlib=3.4.2	


    2. 학습 모드 설정 (Train_network_anomaly_detection.ipynb)
        2.1	ESoinn 선언 및 hyper pram 설정
            1)	iteration_threshold : classify 하기 위한 입력 signal 갯수, 일종의 batch size
            2)	max_edge_age : 학습 중 오래된 edge를 제거하기 위한 threshold
        
        2.2	Check_point 설정
            1)	save_version : 모델 버전
            2)	monitor : 학습 평가값 (eg. accuracy, precison, recall, f1_score)
            3)	save_best_only : 이전 학습 모델 보다 더 나은 monitor값을 갖는 모델만 저장
            4)	save_plt_fig : 학습 과정의 matplot 생성 유무
            5)	patience : 더이상 monitor 값의 개선이 없는 경우, 학습을 Early Stopping하는 기준
        
        2.3	Fit 설정 및 학습 시작
            1)	train_data : 학습을 위한 데이터
            2)	validation_data : 학습 중, 매 epoch( 1 Iteration_threshold) 마다 검증하기 위한 데이터
            3)	epochs : 학습 횟수
            4)	full_shuffle_flag : 모든 입력 데이터를 shuffle 해서 학습을 할것인지(default : True), Bagging 방식으로 학습할 것인지(False) 선택
            [참고] Bagging (Bootstrap Aggregating) : 학습 데이터의 다양성을 위해 무작위로 복원 추출하여 데이터를 추출하는 방식
        
        2.4	학습 모델 저장
            1)	esoinn_{모델 버전}_{epoch 횟수}.pickle 파일명으로 binary 형태 객체 저장


    3. 예측 모드 설정 (AI_engine_performance.ipynb)
        3.1	학습된 모델 load
            1)	모델 버전과 epoch 값을 받아 학습된 모델을 load 한다.
        3.2	예측 실행
            1)	데이터를 입력 받아, 공격 데이터와 일반 데이터의 index를 list로 반환한다.
        3.3	모델 평가
            1)	공격과 일반(‘normal’)에 대한 입력 데이터의 라벨이 있는 경우, 예측 모델의 accuracy, precison, recall, f1_score를 출력한다.
