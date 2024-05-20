# 자연어 처리를 활용한 와인 추천 챗봇 구현

## Team
 - 윤보라
 - 김보경
 - 김경현
 - 김민성

## 프로젝트 개요
 - SVM알고리즘 기반의 SVC모델을 활용한 데이터 전처리
 - TF-IDF기법과 코사인 유사도 기법을 활용한 자연어 처리
 - 파이썬 Streamlit 라이브러리를 활용한 챗봇 구현

### 프로젝트 소개
#### 주제 선정 이유
 - 와인은 사회적 모임, 축하 행사, 특별한 자리에서 자주 즐겨지는 음료입니다. 와인은 사람들을 모이게 하고, 대화를 촉진하며, 특별한 순간을 더욱 특별하게 만듭니다. 와인을 함께 마시는 것은 문화적 경험이자 사회적 활동이 됩니다. 와인에 대해 첫 걸음을 떼시는 분들 혹은 다양한 와인을 경험하고자 하시는 분들께 도움이 되고자 이 주제를 선정하게 되었습니다.

### 1. 데이터셋 선정

와인 리뷰 데이터
https://www.kaggle.com/datasets/zynicide/wine-reviews

### 2. 컬럼 특성 분석 및 선택

- country : 와인 생산국
- description : 평가
- designation : 와인을 생산한 양조장 안의 포도밭
- points :  와인을 1-100점 척도로 평가한 WineEnthusiast 점수
- price : 가격
- province : 와인을 생산한 주
- region_1 : 와인을 숙성한 주
- region_2 : 와인을 숙성한 주의 더 특정된 구역
- taster_name : 평가자 이름
- taster_twitter_handle : 평가자 트위터 주소
- title : 와인의 이름
- variety : 포도 품종
- winery : 양조장 이름

**2-1. NLP 처리에 사용하기 위해 선택한 컬럼**
사용자의 요구사항과 관련된 정보를 가진 컬럼 선택
- country
- description
- price
- title

### 3. 데이터 시각화

1. Country 칼럼 시각화

```python
import pandas as pd
import matplotlib.pyplot as plt

# df = df['country']

# 국가별 행의 개수 계산
country_counts = df['country'].value_counts()

# 시각화
plt.figure(figsize=(12, 6))
country_counts.plot(kind='bar')
plt.title('Number of Rows per Country')
plt.xlabel('Country')
plt.ylabel('Number of Rows')
plt.xticks(rotation=45, ha='right')  # X 축 레이블 회전
plt.tight_layout()
plt.show()
```
![country](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/1424134a-744f-403d-bce5-92cbbcea2a13)

2. 단어 출현 빈도 Top50 시각화

```python
# 단어 출현 빈도 탑50 시각화

from collections import Counter
import matplotlib.pyplot as plt

stopwords = nltk.corpus.stopwords.words('english')

# WordNetLemmatizer 초기화
lemmatizer = WordNetLemmatizer()

# WordNet 품사 태그를 NLTK 품사 태그로 매핑
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # 디폴트는 명사

# 전처리 및 토큰화 함수 정의
def preprocess_and_tokenize(text):
    # 문장 토큰화
    sentences = sent_tokenize(text)
    filtered_words = []

    for sentence in sentences:
        # 단어 토큰화 및 소문자 변환
        words = word_tokenize(sentence.lower())
        for word in words:
            # 불용어 제거 및 특정 단어 제외
            if word not in stopwords and word not in ['wine','flavor', "'s", '%', 'make', 'offer', 'give','.']:  # 여러 개의 단어 제외
                # Lemmatization 수행
                lemmatized_word = lemmatizer.lemmatize(word, get_wordnet_pos(word))
                # 특정 문자 제외
                if lemmatized_word != ',':
                    filtered_words.append(lemmatized_word)

    return filtered_words

# 전처리된 단어 토큰 추출
word_tokens = df['description'].apply(preprocess_and_tokenize)

# 전체 단어 토큰을 하나의 리스트로 펼치기
all_words = [word for sublist in word_tokens for word in sublist]

# 단어 빈도 계산
word_freq = Counter(all_words)

# 가장 빈도가 높은 단어 순으로 정렬
word_freq_sorted = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))

# 상위 50개의 단어만 선택
top_words = dict(list(word_freq_sorted.items())[:50])

# 시각화
plt.figure(figsize=(12, 6))
plt.bar(top_words.keys(), top_words.values())
plt.title('Top 50 Word Frequency in Descriptions')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')  # X 축 레이블 회전
plt.tight_layout()
plt.show()
```
![description](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/50ae8f6e-008c-4314-9731-188a421b35cd)


3. Price 컬럼 시각화

```python
import pandas as pd
import matplotlib.pyplot as plt

# 가격별 행의 개수 계산 및 오름차순 정렬
price_counts = df['price'].value_counts().sort_index()

# 시각화
plt.figure(figsize=(12, 6))
country_counts.plot(kind='bar')
plt.title('Number of Rows per Price')
plt.xlabel('Price')
plt.ylabel('Number of Rows')

# x 축 눈금 설정 (원하는 간격으로)
tick_spacing = 20  # 간격 설정
plt.xticks(range(0, len(country_counts), tick_spacing), country_counts.index[::tick_spacing], rotation=0, ha='right')

plt.tight_layout()
plt.show()
```
![price](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/f0788af7-4284-4c2b-ba09-7db317a4d72e)


4. Variety Top50 시각화

```python
# Variety 탑50 시각화
import pandas as pd
import matplotlib.pyplot as plt

# 종류별 행의 개수 계산 및 오름차순 정렬
variety_counts = df['variety'].value_counts().head(50)

# 시각화
plt.figure(figsize=(12, 6))
variety_counts.plot(kind='bar')
plt.title('Number of Rows per variety top 50')
plt.xlabel('Variety')
plt.ylabel('Number of Rows')

# x 축 눈금 설정 (원하는 간격으로)
tick_spacing = 1  # 간격 설정
plt.xticks(range(0, len(variety_counts), tick_spacing), variety_counts.index[::tick_spacing], rotation=45, ha='right')

plt.tight_layout()
plt.show()
```
![variety](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/b2d92db5-f3ac-4e04-9528-ecf970c130b3)



### 4. 결측치 처리

```python
# 데이터셋 불러오기
import pandas as pd
df1 = pd.read_csv('/content/drive/MyDrive/엔코어/winemag-data-130k-v2.csv')
df1
```

![Untitled (28)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/1f163ad3-cc55-4a28-8f67-deca43dd22ee)

```python
# 전체데이터 결측치 확인
df1.isna().sum()
```

![Untitled (1)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/abf1bf2b-3687-4578-9b31-bb17b5193c74)


```python
# 사용할 데이터 남기고 컬럼 드랍하기

df1.drop(columns='Unnamed: 0	', inplace = True)
df1.drop(columns='region_1', inplace = True)
df1.drop(columns='region_2', inplace = True)
df1.drop(columns='taster_twitter_handle', inplace = True)
df1.drop(columns='taster_name', inplace = True)
df1.drop(columns='designation', inplace = True)
```

![Untitled (2)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/ac4272d4-473f-4268-b078-ba26edd0efa7)



winery와 title의 결측치가 없었기 때문에 이 컬럼을 활용해서 country와 province의 결측치를 채워주기

```python
len(df1[df1['country'].isna()]), len(df1[df1['province'].isna()]), len(df1[df1['winery'].isna()])
```

![Untitled (3)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/3de781a5-f775-45e6-a24c-dc2d7086e521)



결측치가 있는 행의 winery 값을 추출

```python
df2[df2['winery'] == 'Gotsa Family Wines']
```

![Untitled (4)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/15ae86c5-9ed4-4ef6-99d0-2f7084ef9303)



winery값을 기반으로 country와 province 값 채워주기

```python
mask = df2['winery'] == 'Gotsa Family Wines'
df2.loc[mask, 'country'] = 'Georgia'
df2.loc[mask, 'province'] = 'Georgia'
df2[df2['winery'] == 'Gotsa Family Wines']
```

![Untitled (5)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/2b318666-68d7-4ae3-9a0d-929c6018a82f)


위의 과정 반복으로 country와 province결측치 처리

※ 가격 결측치는 데이터에서 제외

```python
df2.isna().sum()
```
![Untitled (6)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/47284469-d226-4a44-a2c3-dd94e9bc9be1)


### 5. wine type구분

‘variety’, ‘title’의 특성을 추출해서 각 와인의 타입 구분

```python
# 와인의 종류를 결정하는 함수
def determine_wine_type(row):
    variety = str(row.get("variety", "")).lower()
    title = str(row.get("title", "")).lower()
    # Sparkling Wine 조건 추가
    if any(keyword in variety for keyword in ['spark', 'brut', 'prosecco', 'champagne']) or \
         any(keyword in title for keyword in ["sparkling", "prosecco", "champagne"]):
        return "Sparkling"
    # Red Wine 조건 추가
    elif any(keyword in variety for keyword in ['cabernet sauvignon', 'merlot', 'pinot noir', 'syrah', 'malbec', 'tempranillo', 'sangiovese',
                     'nebbiolo', 'grenache', 'zinfandel', 'barbera', 'petite sirah', 'carmenere', 'petit verdot',
                     'nero d\'avola', 'montepulciano', 'aglianico', 'tannat', 'pinotage', 'dolcetto', 'bonarda',
                     'frappato', 'cesanese', 'negroamaro', 'nerello mascalese', 'sagrantino', 'lagrein', 'primitivo',
                     'mourvèdre', 'carignan', 'touriga nacional', 'tinta roriz', 'garnacha', 'monastrell', 'bobal',
                     'xinomavro', 'agiorgitiko', 'corvina', 'rondinella', 'molinara', 'teroldego', 'refosco', 'schiava',
                     'raboso', 'cesanese d\'affile', 'picpoul', 'gaglioppo', 'nerello cappuccio', 'cannonau', 'pugnitello',
                     'garnacha tintorera', 'graciano', 'tinta miúda', 'bastardo', 'alicante bouschet', 'rufete',
                     'tinta del toro', 'corvina', 'trincadeira', 'prieto picudo', 'tempranillo blend', 'meritage',
                     'red blend', 'rhône-style red blend', 'bordeaux-style red blend', 'cabernet blend',
                     'garnacha blend', 'grenache blend' ]) or \
       any(keyword in title for keyword in ["red", "noir", "merlot", "cabernet", "shiraz"]):
        return "Red"
    # White Wine 조건 추가
    elif any(keyword in variety for keyword in ['chardonnay', 'sauvignon blanc', 'riesling', 'pinot gris', 'pinot grigio', 'gewürztraminer',
                       'viognier', 'chenin blanc', 'albariño', 'grüner veltliner', 'verdejo', 'vermentino',
                       'marsanne', 'roussanne', 'grenache blanc', 'picpoul', 'melon', 'torrontés', 'moscato', 'muscat',
                       'muscat blanc à petits grains', 'fumé blanc', 'müller-thurgau', 'silvaner', 'semillon', 'cortese',
                       'fiano', 'garganega', 'verdicchio', 'grecheetto', 'friulano', 'ribolla gialla', 'kerner', 'sylvaner',
                       'welschriesling', 'scheurebe', 'siegerrebe', 'riesling', 'johannisberg riesling', 'savagnin',
                       'furmint', 'malvasia', 'assyrtiko', 'aidani', 'roditis', 'malagousia', 'robola', 'moschofilero',
                       'white riesling', 'muscat ottonel', 'muskat', 'gelber muskateller', 'sauvignonasse', 'arneis',
                       'erbaluce', 'favorita', 'grechetto', 'pigato', 'timorasso', 'vernaccia', 'verduzzo', 'vermentino nero',
                       'ribolla gialla', 'früburgunder', 'weissburgunder', 'morillon', 'roter veltliner', 'zierfandler',
                       'rotgipfler', 'neuburger', 'blanc du bois', 'chenin blanc-chardonnay', 'chenin blanc-sauvignon blanc',
                       'alsace white blend', 'rhône-style white blend', 'bordeaux-style white blend', 'white blend',
                       'austrian white blend']) or \
         any(keyword in title for keyword in ["white", "gris", "chardonnay", "sauvignon", "pinot blanc"]):
        return "White"
    # Rosé Wine 조건 추가
    elif any(keyword in variety for keyword in ["rosé", "rose", "rosato", "rosado"]) or \
         any(keyword in title for keyword in ["rosé", "rose"]):
        return "Rosé"
    else:
        return "Unknown"
# 'wine_type' 컬럼 생성
df["wine_type"] = df.apply(determine_wine_type, axis=1)
df.head()
```
![Untitled (7)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/50f30a92-ac05-470d-b6f2-7e67cdc6f73c)


```python
df['wine_type'].value_counts()
```

![Untitled (8)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/25914b26-b36b-47b1-a37d-6c70bd544f62)


와인 type을 예측하지 못한 Unknown은 제거

```python
df_sorted = df_unique.sort_values(by=['title', 'price'])
unique_title_rows = df_sorted.drop_duplicates(subset=['title'], keep='first')
df_filtered  = unique_title_rows[unique_title_rows['wine_type'] != 'Unknown']
```

### **6. 머신러닝 학습을 통한 Wine Type 추출**

1. Tf-idf와 LogisticRegression을 이용해 학습한 모델로 와인 타입 추측

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 데이터셋 로드
df = pd.read_csv('/content/drive/MyDrive/엔코어/my_wine_data.csv')

# 결측치 제거
df.dropna(inplace=True)

# 텍스트 데이터 결합
df['text_combined'] = df['description'] + ' ' + df['title'] + ' ' + df['variety']

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['text_combined'])

# 분류 모델 학습
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X, df['wine_type'])

# 예측
df['predicted_wine_type'] = logreg.predict(X)

# 추가 조건에 따라 예측값 수정
df.loc[df['text_combined'].str.contains('spark', case=False), 'predicted_wine_type'] = 'Sparkling'
# df.loc[df['text_combined'].str.contains('rose', case=False), 'predicted_wine_type'] = 'Rose'

# 결과 확인
print(df[['text_combined', 'predicted_wine_type']])
```

![Untitled (9)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/abf0e803-aad9-4993-82a9-77fd52a0ed7c)


```python
# 예측한 값과 실제 값과 일치하는 비율 확인
matching_rows = df[df['predicted_wine_type'] == df['wine_type']]
matching_ratio = len(matching_rows) / len(df) * 100

print(f"예측한 와인 타입과 실제 와인 타입이 일치하는 비율: {matching_ratio:.2f}%")
```

![Untitled (10)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/5ab1d15b-6c61-4cc3-beb0-96b9f97ceeda)


```python
# predicted_wine이랑 wine_type이랑 비교 시각화

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 혼동 행렬 생성
cm = confusion_matrix(df['wine_type'], df['predicted_wine_type'])

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df['wine_type'].unique()),
            yticklabels=sorted(df['wine_type'].unique()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```
![Untitled (11)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/8604b70c-8d50-48fd-a178-0624c3224326)



2.  Keras의 딥러닝을 이용해 학습한 모델로 와인 타입 예측

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# 데이터셋 로드
df = pd.read_csv('/content/drive/MyDrive/엔코어/my_wine_data.csv')
df['text_combined'] = df['description'] + ' ' + df['title'] + ' ' + df['variety']

# 입력과 타겟 데이터 설정
X = df['text_combined']
y = df['wine_type']

# 토크나이저 설정
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)

# 텍스트를 시퀀스로 변환
X_seq = tokenizer.texts_to_sequences(X)

# 시퀀스 길이를 맞춰주기 위해 패딩 추가
X_pad = pad_sequences(X_seq, maxlen=100)

# 타겟 데이터 원-핫 인코딩
y_encoded = pd.get_dummies(y)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_encoded, test_size=0.2, random_state=42)

# LSTM 모델 정의
model = Sequential()
model.add(Embedding(5000, 128, input_length=100))
model.add(LSTM(128))
model.add(Dense(4, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64)

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
```

![Untitled (12)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/c02dd2ea-9c30-4837-a39f-fec278cfcc25)


```python
# 학습시킨 모델로 전체 데이터를 예측해본 결과
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

# 데이터셋 로드
df = pd.read_csv('/content/drive/MyDrive/엔코어/my_wine_data.csv')
df['text_combined'] = df['description'] + ' ' + df['title'] + ' ' + df['variety']

# 토크나이저 설정
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['text_combined'])

# 텍스트를 시퀀스로 변환
X_seq = tokenizer.texts_to_sequences(df['text_combined'])

# 시퀀스 길이를 맞춰주기 위해 패딩 추가
X_pad = pad_sequences(X_seq, maxlen=100)

# 모델을 사용하여 예측 수행
predictions = model.predict(X_pad)

# 예측 결과를 분석하여 필요한 처리를 수행합니다.
# 여기서는 각 예측 값의 인덱스를 사용하여 해당하는 와인 타입을 찾을 수 있습니다.

# 예측 결과를 DataFrame에 추가
df['predicted_wine_type'] = predictions.argmax(axis=1)

# 결과를 저장하거나 출력합니다.
print(df[['text_combined', 'predicted_wine_type']])
```

![Untitled (13)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/e1abb22d-bc20-4f2f-b3f4-166ec2887a28)


```python
# 원핫 인코딩 결과로 출력된걸 각 번호에 맞는 와인으로 변경
wine_type_map = {0: 'Red', 1: 'Sparkling', 2: 'Rose', 3: 'White'}

# 예측 결과를 와인 타입으로 변환하여 DataFrame에 추가
df['predicted_wine_type'] = df['predicted_wine_type'].map(wine_type_map)

# 결과를 출력합니다.
print(df[['text_combined', 'predicted_wine_type']])
```

![Untitled (14)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/aadd2186-caca-437b-90c0-af2495017ffd)


```python
# 예측한 값과 실제 값과 일치하는 비율 확인

matching_rows = df[df['predicted_wine_type'] == df['wine_type']]
matching_ratio = len(matching_rows) / len(df) * 100

print(f"예측한 와인 타입과 실제 와인 타입이 일치하는 비율: {matching_ratio:.2f}%")
```

![Untitled (15)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/3171803b-baa0-4ea8-8108-502757d98910)


```python
# predicted_wine이랑 wine_type이랑 비교 시각화

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 혼동 행렬 생성
cm = confusion_matrix(df['wine_type'], df['predicted_wine_type'])

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df['wine_type'].unique()),
            yticklabels=sorted(df['wine_type'].unique()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

![Untitled (16)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/42b5843b-71a9-413b-8cc1-83346db43a60)


1.   Tf-idf와 SVC를 이용해 학습한 모델로 와인 타입 예측

```python
# SVC를 이용해서 분할 해서 test돌린 결과값 : (Accuracy: 0.9889141425854632)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 데이터셋 로드
df = pd.read_csv('/content/drive/MyDrive/엔코어/wine_data_with_predictions_v1.csv')
df['text_combined'] = df['description'] + ' ' + df['title'] + ' ' + df['variety']

# 입력과 타겟 데이터 설정
X = df['text_combined']
y = df['wine_type']

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# SVM 모델 정의 및 학습
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# 테스트 세트에서 예측
y_pred = svm_model.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
![Untitled (17)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/a5c12210-fff2-4ba3-a5c7-3b68bd3e1799)



```python
# 1차로 나온 predicted_wine이랑 wine_type이랑 비교 시각화

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 혼동 행렬 생성
cm = confusion_matrix(df['wine_type'], df['predicted_wine_type'])

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df['wine_type'].unique()),
            yticklabels=sorted(df['wine_type'].unique()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

![Untitled (18)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/eb8abfa0-ec79-4bde-bad2-8f209f47c6ba)


```python
# 1차로 학습시킨 결과값을 이용해서 다시 전체를돌린 결과값 : (Accuracy: 0.9943318335729857)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 데이터셋 로드
df = pd.read_csv('/content/drive/MyDrive/엔코어/wine_data_with_predictions_v1.csv')
df['text_combined'] = df['description'] + ' ' + df['title'] + ' ' + df['variety']

# 입력과 타겟 데이터 설정
X = df['text_combined']
y = df['wine_type']

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# SVM 모델 정의 및 학습
svm_model = SVC(kernel='linear')
svm_model.fit(X_tfidf, y)

# 전체 데이터에 대한 예측
y_pred = svm_model.predict(X_tfidf)

# 정확도 계산
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)
```
![Untitled (19)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/a7148759-23a5-4515-8d4a-a344f09e23ed)


```python
# 1차로 나온 predicted_wine이랑 wine_type이랑 비교 시각화

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 혼동 행렬 생성
cm = confusion_matrix(df['wine_type'], df['predicted_wine_type'])

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df['wine_type'].unique()),
            yticklabels=sorted(df['wine_type'].unique()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```
![Untitled (20)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/7655c3cf-1304-40ea-9c45-0b203081bc20)


 **모델 선택**

임베딩, LSTM, 밀집 레이어 -> 비정형 데이터를 다루는 딥러닝에 사용된다.

LogisticRegression -> 단순하고 빠르지만 복잡한 패턴에는 한계가 있다.

SVC -> 선형 및 비선형 문제에 모두 적합하지만 큰 데이터셋에서는 속도가 느리다.

임베딩, LSTM, 밀집 레이어 -> 복합한 데이터 구조와 종속성을 캡처하데 매우 효과적이나 매우 복잡하고 계산 비용이 크다.

예측 테이블을 정답이라고 가정하고 진행한 지도학습이기 때문에 정확성이 높은것이 정답이라고 결정하긴 어렵지만, 이중 가장 적합하다고 판단한 SVC모델로 예측한 wine_type을 정답이라고 가정하고 진행했다.

이후 같은 모델로 여러번 학습을 진행시켜 오차값 비교 후 수정을 통해 오답률을 최소화했다.

```python
# 2차로 학습시킨 결과값을 이용해서 다시 전체를돌린 결과값 : (Accuracy: 0.9987012384618953)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 데이터셋 로드
df = pd.read_csv('/content/drive/MyDrive/엔코어/wine_data_with_predictions_v3.csv')
df['text_combined'] = df['description'] + ' ' + df['title'] + ' ' + df['variety']

# 입력과 타겟 데이터 설정
X = df['text_combined']
y = df['wine_type']

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# SVM 모델 정의 및 학습
svm_model = SVC(kernel='linear')
svm_model.fit(X_tfidf, y)

# 전체 데이터에 대한 예측
y_pred = svm_model.predict(X_tfidf)

# 정확도 계산
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)
```
![Untitled (21)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/6456c63d-99e8-47ad-bd88-5d6b4968bac9)


```python
# 2차로 나온 predicted_wine이랑 wine_type이랑 비교 시각화

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 혼동 행렬 생성
cm = confusion_matrix(df['wine_type'], df['predicted_wine_type'])

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df['wine_type'].unique()),
            yticklabels=sorted(df['wine_type'].unique()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

![Untitled (22)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/4199292c-07e4-4e34-9be8-53d84bdc074e)


```python
# 3차로 학습시킨 결과값을 이용해서 다시 전체를돌린 결과값 : (Accuracy: 0.9993227886265597)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 데이터셋 로드
df = pd.read_csv('/content/wine_data_with_predictions_v4.csv')
df['text_combined'] = df['description'] + ' ' + df['title'] + ' ' + df['variety']

# 입력과 타겟 데이터 설정
X = df['text_combined']
y = df['wine_type']

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# SVM 모델 정의 및 학습
svm_model = SVC(kernel='linear')
svm_model.fit(X_tfidf, y)

# 전체 데이터에 대한 예측
y_pred = svm_model.predict(X_tfidf)

# 정확도 계산
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)
```

![Untitled (23)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/fc42b509-53c0-47e4-adc4-d4230a912ae9)


```python
# 3차로 나온 predicted_wine이랑 wine_type이랑 비교 시각화

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 혼동 행렬 생성
cm = confusion_matrix(df['wine_type'], df['predicted_wine_type'])

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df['wine_type'].unique()),
            yticklabels=sorted(df['wine_type'].unique()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```
![Untitled (24)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/6c1c2fda-a0c4-4a28-945d-438d80904031)


### 7. 챗봇 구현

---

**Streamlit 라이브러리**

- 데이터 과학 및 머신러닝 애플리케이션을 만들기 위한 웹 애플리케이션 프레임워크

**챗봇 브리핑**

- 데이터와 사용자 입력값의 유사도 판단
    
    **TF-IDF** 
    
    - TF : 특정 단어가 문서 내에서 얼마나 자주 나타나는지를 측정한 값. 단어 빈도는 해당 단어가 문서 내에서 등장한 횟수를 해당 문서의 전체 단어 수로 나눈 것이다.
    - IDF : 특정 단어가 얼마나 많은 문서에서 나타나는지를 측정한 값. IDF는 해당 단어가 포함된 문서의 비율의 역수를 취한 로그값으로 정의한 것이다.
    - TF-IDF는 TF와 IDF를 곱한 값으로, 단어의 상대적인 중요성을 측정하는 데 사용되어진다.
    
    **코사인 유사도**
    
    - 벡터 공간에서 두 벡터의 유사성을 측정하는 방법 중 하나
    - 텍스트 데이터에서 유사성을 평가하는 데에 효과적이며, 자카드 지수나 유클리드 거리보다 어휘 유사성을 더 잘 고려하여 일반적으로 텍스트 데이터에 대한 유사성 작업에서는 코사인 유사도가 선호 된다.
    
    similarity 컬럼을 추가하여 유사도 수치를 확인
    
![Untitled (25)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/512e1479-9f7a-488d-834c-f28292c2cba8)

    
- wine_type 필터링
    
    사용자가 와인 타입에 해당하는 키워드를 입력하면 해당 타입의 와인들을 추천받을 수 있다.
    

    ![Untitled (26)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/b77c8b71-514e-4b4c-a66b-527986659f7d)


- price 필터링
    
    사용자가 ["price", "cost", "priced", "dollar", "dollars", "$"] 등의 단어를 입력하면 나오는 사이드 바에 가격대의 범위를 지정할 수 있다.
    
![Untitled (27)](https://github.com/pladata-encore/DE30-2nd-4/assets/163945173/6a779654-8dd9-46a1-a120-a0b29594d217)

    

 **배포**
- github repository 생성 https://github.com/
    
- 파이썬 파일과 데이터셋파일 업로드
    
- streamlit 홈페이지에서 회원가입 및 git 계정과 연동 진행 https://share.streamlit.io/
- create app을 눌러 배포 진행
- https://wine-recommender-chatbot.streamlit.app/

### 8. 결론
- 사용자 입력값에서 수치형 값이 가격대를 의미하는지 생산연도를 의미하는지 파악에 어려움이 있다.
- 와인의 타입별 분류에서 소단위 분류를 통해 더 자세한 분류가 가능하게 지속적인 개선이 필요하다.
