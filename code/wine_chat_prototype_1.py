import pandas as pd
import random
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

df = pd.read_csv('wine_prototype_1.csv')

def extract_numbers(input_string):
    # 정규식을 사용하여 입력에서 숫자 추출
    numbers = re.findall(r'\d+', input_string)
    return [int(num) for num in numbers]

def extract_number(input_string):
    # 입력에서 숫자 추출
    match = re.search(r'\d+', input_string)
    if match:
        return int(match.group())
    else:
        return None

def filter_range(input_string):
    # 입력에서 숫자 추출
    numbers = extract_numbers(input_string)
    # 숫자가 2개인지 확인
    if len(numbers) == 2:
        # 두 숫자 중 작은 값과 큰 값을 상한값과 하한값으로 설정
        lower_bound = min(numbers)
        upper_bound = max(numbers)
        return lower_bound, upper_bound
    else:
        return filter_values(input_string)

def filter_values(input_string):
    # 입력 문자열을 소문자로 변환
    input_lower = input_string.lower()
    # 입력에서 숫자 추출
    number = extract_number(input_string)
    # over, upper, up이 있는 경우
    if 'over' in input_lower or 'upper' in input_lower or 'up' in input_lower or 'above' in input_lower:
        return number if number is not None else None, float('inf')
    # lower, low, down이 있는 경우
    elif 'lower' in input_lower or 'low' in input_lower or 'down' in input_lower or 'under' in input_lower:
        return float('-inf'), number if number is not None else None
    # 해당 단어가 없는 경우
    else:
        return None, None

def extract_country_from_list(input_string, value_list):
  input_words = input_string.lower().split()
  extracted_values = []
  for word in input_words:
    if word.endswith(".") or word.endswith(",") or word.endswith("?"):
      word = word[:-1]
    if word == 'usa' or word == 'america':
      word = 'us'
    if word in value_list:
      extracted_values.append(word)
  return [i.capitalize() if i != 'us' else i.upper() for i in extracted_values] if extracted_values else None

def extract_wine_type_from_list(input_string, value_list):
    input_words = input_string.lower().split()
    extracted_wine_type = []
    for word in input_words:
      if word.endswith(".") or word.endswith(",") or word.endswith("?"):
        word = word[:-1]
      if word in value_list:
          extracted_wine_type.append(word)
    return [i.capitalize() for i in extracted_wine_type] if extracted_wine_type else None


def find_similar_rows(user_input, n=5):
    # 입력 문장을 TF-IDF로 벡터화
    sentence_tfidf = tfidf_vectorizer.transform([user_input])

    # 입력 문장과 각 행 간의 코사인 유사도 계산
    similarities = cosine_similarity(sentence_tfidf, tfidf_matrix)

    # 가장 유사한 행의 인덱스 찾기 (상위 n개)
    similar_indices = similarities.argsort()[0][::-1]
    # 유사한 행 출력
    similar_rows = filtered_df.iloc[similar_indices]

    return similar_rows


# 초기에는 전체 데이터를 담은 DataFrame을 사용합니다.
filtered_df = df.copy()
unique_wine_type = [wine_type.lower() for wine_type in df['wine_type'].unique().tolist()]
unique_price = [price for price in df['price'].unique().tolist()]
unique_countries = [country.lower() for country in df['country'].unique().tolist()]
top_varieties = df['variety'].value_counts().head(10).index.tolist()
top_10_prices = df['price'].value_counts().head(10).index.tolist()

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df[['description', 'title']].apply(lambda x: ' '.join(x), axis=1))

# 필터링을 여러 번 반복합니다.
# Streamlit 앱 생성
st.title('와인 발사대')
input_value = st.text_input('You:',key='first')
st.button('Commit')
while True:
    if input_value.lower() == 'done':
      break
    if input_value.lower() == 'clear':
      break
    if 'anything' in input_value:
      st.write('''Let me recommend some of the popular wines here. If you want other wines, I can recommend more specific wines by typing in some of the keywords below.''')
      random_numbers = [random.sample(range(1, len(df.copy())), 10)]

      filtered_df = df[df['price'].isin(top_10_prices) & df['variety'].isin(top_varieties)]
      random_5_rows = filtered_df.sample(n=5)
      st.write(random_5_rows[['country','title','description','price','wine_type']])
      st.subheader('Keyword')
      st.write('''Variety-based search: search by variety of wine (e.g. chardonnay, pinot noir, etc.).\n
Country-based search: Search by country from which wine was produced (e.g. US, France, etc.).\n
Type-based search: search by color of wine (e.g. red, white, rosé, sparkling).\n
Price-based search: You can specify a range of price points you want or search by numbers that contain the keywords 'above' or 'below'.''')
      filtered_df = df.copy()
      break
    try:
      filtered_df = find_similar_rows(input_value, n=5)
    except:
        pass
        
    for word in input_value.split():
      word = word.lower()
      if word.endswith(".") or word.endswith(",") or word.endswith("?"):
          word = word[:-1]
      if word == 'usa' or word == 'america':
          word = 'us'
      if word in unique_countries :
        extracted_countries = extract_country_from_list(input_value, unique_countries)
        country_word = extracted_countries
        filtered_df = filtered_df[filtered_df['country'].isin(extracted_countries)]

      if word in unique_wine_type:
        extracted_wine_type = extract_wine_type_from_list(input_value, unique_wine_type)
        filtered_df = filtered_df[filtered_df['wine_type'].isin(extracted_wine_type)]
      
      if 'most' in word:
        filtered_df = filtered_df[filtered_df['price'] == filtered_df['price'].max()]
      
      if 'least' in word or 'cheapest' in word:
        filtered_df = filtered_df[filtered_df['price'] == filtered_df['price'].min()]

      if word.endswith('$'):
        word = word[:-1]
        if int(word) in unique_price:
          numbers = extract_numbers(input_value)
          if len(numbers) == 2:
            lower_bound = min(numbers)
            upper_bound = max(numbers)
            filtered_range = filter_range(input_value)
            filtered_df = filtered_df[(filtered_df['price'] >= lower_bound) & (filtered_df['price'] <= upper_bound)]
          elif len(numbers) == 1:
            min_value, max_value = filter_values(input_value)

            if 'over' in input_value.lower() or 'upper' in input_value.lower() or 'up' in input_value.lower() or 'above' in input_value.lower():
              filtered_df = filtered_df[filtered_df['price'] >= min_value]

            elif 'lower' in input_value.lower() or 'low' in input_value.lower() or 'down' in input_value.lower() or 'under' in input_value.lower():
              filtered_df = filtered_df[filtered_df['price'] <= max_value]
      elif word.startswith('$'):
        word = word[1:]
        if int(word) in unique_price:
          numbers = extract_numbers(input_value)
          if len(numbers) == 2:
            lower_bound = min(numbers)
            upper_bound = max(numbers)
            filtered_range = filter_range(input_value)
            filtered_df = filtered_df[(filtered_df['price'] >= lower_bound) & (filtered_df['price'] <= upper_bound)]
          elif len(numbers) == 1:
            min_value, max_value = filter_values(input_value)

            if 'over' in input_value.lower() or 'upper' in input_value.lower() or 'up' in input_value.lower() or 'above' in input_value.lower():
              filtered_df = filtered_df[filtered_df['price'] >= min_value]

            elif 'lower' in input_value.lower() or 'low' in input_value.lower() or 'down' in input_value.lower() or 'under' in input_value.lower():
              filtered_df = filtered_df[filtered_df['price'] <= max_value]
    # print(f'total {len(filtered_df)} searched')
    # print(filtered_df[['country','title','description','price','wine_type']].head(5))
    st.text('발사대:')
    st.write(f'total {len(filtered_df)} searched')
    st.write('-'*20)
    count=0
    if 'show all' in input_value:
        st.write(filtered_df[['title','description','price','wine_type','country']])
    for index, row in filtered_df.iterrows():
        st.write(f"Title: {row['title']}")
        st.write(f"Description: \n{row['description']}")
        st.write(f"Price: {row['price']}$")
        st.write(f"Type: {row['wine_type']}")
        st.write(f"Country: {row['country']}")
        st.write("----")
        count+=1
        if count ==10:
            break
    break
