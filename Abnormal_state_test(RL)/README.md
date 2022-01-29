# abnormal state model
# 평균 성능 Accuracy 약 97% (3회 측정)
    
    KISA 검증       

    1. 실행 방법 (abnormal_state_model.py)
        1.1 학습 & 예측 데이터 그래프화
            1) abnormal_state 디렉토리 내 create_tda_graph.ipynb 노트북 파일 코드 활용, 학습 및 예측 데이터 그래프화
    
        1.2 Terminal에서 해당 py 파일 실행
            1) python3.7 abnormal_state_model.py       
            
        1.3 모드 설정
            1) 라이브러리 & config 파일 로드 후 모드 입력
                1-1) <<Enter the mode>> 출력 후 모드 입력
                1-2) 학습일 경우 train 입력 & 예측일 경우 predict 입력
    
    2. 데이터 그래프화 (학습 & 예측 동일 프로세스)
        2.1 라이브러리 & config 로드
            1) 모델 이름, 저장 버전, 결과 저장되는 DB 테이블 등에 관한 정보 로드
                1-1) 해당 정보들에 대한 수정 필요 시 DTI > AI Model Setting에서 수정, 또는 노트북 코드 내에서 config 로드 후 직접 수정
            
        2.2 데이터 로드
            1) 디렉토리 내 data.py 파일의 DataCreation 클래스를 통해 데이터 로드
                1-1) 디렉토리 내 query.py의 쿼리를 사용하여 데이터 파싱
            2) 데이터 기간, 공격 유형은 data.py 파일의 __init__ 부분에서 확인 & 수정 가능
                                     
            
        2.3 데이터 전처리
            1) 디렉토리 내 prep.py 파일의 StrProcessing 클래스를 통해 문자영 데이터 전처리 진행
                1-1) TF-IDF 사용
                1-2) TF-IDF fitting 후 pickle 형식으로 자동 저장됨
                1-3) 저장 경로 확인 & 수정은 prep.py의 StrProcessing 클래스 내 save_tfidf_model_fit, load_tfidf_model_trans 함수에서 확인 & 수정 가능
            
        2.4 라벨 변환
            1) XSS, CREDENTIAL 등으로 라벨되어 있는 데이터를 해당 공격에 대한 대응방안으로 재 라벨링
        
        2.5 데이터 셔플 함수 생성
            1) 데이터 100 raw를 하나의 네트워크 상태로 가정하고 라벨이 정상일 시 정상 데이터 100 raw만,
               공격 라벨일 경우 정상 50 raw + 공격 50 raw 사용 (데이터는 무작위 추출)
            2) raw 갯수의 수정을 원할 경우 create_shuffle_data 함수 내 n 값을 변경
        
        2.6 KeplerMapper & Stellar graph 라이브러리 로드
            1) 데이터 그래프화에 활용될 TDA (KeplerMapper)와 Stellar Graph 라이브러리 로드
        
        2.7 데이터 변환
            1) create_shuffle_data 함수를 사용하여 만든 하나의 네트워크 상태(100 raw)에 대해 그래프화 진행
                1-1) 각 라벨별 800개의 그래프 데이터 리스트를 생성하도록 구성되어 있으며, 그래프 리스트 갯수 수정을 원할 시 for loof in range(800)에서 range 값 조정
            2) 데이터 차원 축소 (PCA)
                2-1) 다른 기법 사용 원할 시 mapper.fit_transform 내 projection에서 수정 가능 (하이퍼파라미터에 대한 자세한 정보는 Sklearn의 Keplermapper Document 참조)
            3) 노드, 엣지, 차원 축소 데이터, 메타정보 활용 그래프 생성
                3-1) 차원 축소된 lens 데이터를 기반으로 hypercube를 생성, hypercube별 인덱스를 추출, 추출된 인덱스를 원본데이터에서 조회, hypercube별(원본) cluster 진행
                3-2) Cluster 기법은 K-means를 사용, 다른 기법 사용 원할 시 mapper.map내 clusterer에서 수정 가능
            4) Stellar graph & Data Save
                4-1) 그래프 데이터 학습을 위해 KeplerMapper로 생성한 그래프 이미지 Stellar graph화 진행
                4-2) 100 raw로 생성한 하나의 네트워크 상태를 10번 List에 쌓아 하나의 pickle 파일로 저장 (1000 raw가 하나의 pickle)
                4-3) 학습 데이터 검증 & 예측 후 결과값 저장을 위해 그래프에 대한 원본 데이터 Graph list pickle 파일에 함께 저장

    3. 학습 모드
        3.1 학습 데이터 로드
            1) 그래프화된 학습 데이터 로드 (Graph list pickle 파일)
            
        3.2 generator 생성
        
        3.3 디렉토리 내 RL.model.py의 train_environment Class 활용 학습 환경 구성
            
        3.4 Get actor & critic
            1) RL.model.py의 get_actor, get_critic 함수 사용
                1-1) actor_model, critic_model, target_actor, target_critic 생성
                          
        3.5 모델 학습 (Buffer)
            1) RL.model.py의 Buffer Class 활용, 매 step마다 주어지는 stage, action, Q-Value에 대한 값들, 즉 경험(Experience)이 저장됨
            2) 저장된 경험을 이용해 Network를 학습할 때 사용
            3) total_episodes 변수로 모델 학습 반복 수 (epoch) 조정
            4) 학습 완료 후 actor_model, episodic_reward, labels_dict 지정된 경로에 저장
            
    4. 예측 모드
        4.1 학습된 모델 로드
            1) episodic_reward, labels_dict 로드
            
        4.2 예측 데이터 로드
            1) 그래프화된 예측 데이터 로드 (Graph list pickle 파일)
            
        4.2 generator 생성
        
        4.3 디렉토리 내 RL.model.py의 pred_environment Class 활용 예측 환경 구성
            
        4.4 Actor 모델 로드
            1) actor_model 로드 후 summary 출력
                          
        4.5 모델 예측
            1) 모델 ACCURACY & Counfusion matrix 출력
            2) 디렉토리 내 utils.py 활용 DB(clickhouse) 연결
            3) 예측된 대응방안 원본 데이터에 라벨링 후 DB에 Insert
            