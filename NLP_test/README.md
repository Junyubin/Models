# preprocessing & auto profiling (tf_idf / Word2Vec) 
# 평균 성능 accuracy (tf_idf : 0.97 / Word2Vec 0.96)

    1. 실행 과정
        1.1	정상 & 공격 유형별 데이터 가져오기
            1) 날짜, 개수, grupby 기준 설정 가능
            
        1.2 데이터 분리
            1) 학습 데이터, 예측 데이터 분리
            2) 7:3으로 분리 (변경 가능)
            
        1.3 데이터 전처리
            1) 'host', 'agent', 'query' 컬럼들에 대해 진행
            2) tfidf.ipynb : tf_idf를 통한 데이터 vectorizing
                           : auto_profiling_utils.py 의 DataPreprocessing 클래스 활용
            3) w2v.ipynb : word2vec를 통한 데이터 vectorizing
                         : 단어 length와 vector size 지정 가능
                         
        1.4 학습 실행
            1) auto_profiling_model.py의 cnn 모델 활용
            2) 두 전처리 방법 모두 같은 모델을 활용햇으며, config에서 관련 파라미터 수정 가능
            3) 모델 Fitting 후 h5 형식으로 자동 저장
            
        1.5 예측 실행
            1) 학습 된 모델 호출
            2) 데이터 예측 후 검증 시행
            3) 검증은 Accuracy score와 Confusion Matrix 점수로 확인            
            