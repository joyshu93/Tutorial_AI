import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 간단한 대화 데이터
conversations = [
    ("안녕", "안녕하세요"),
    ("뭐해", "일하고 있어요."),
    ("잘가", "안녕히 가세요!")
]

# 입력과 출력 문장 분리
questions, answers = zip(*conversations)

# 토큰화
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
total_words = len(tokenizer.word_index) + 1

# 문장을 시퀀스로 변환
input_sequences = tokenizer.texts_to_sequences(questions)
output_sequences = tokenizer.texts_to_sequences(answers)

# 패딩
max_sequence_length = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in output_sequences))
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_sequence_length, padding='post')

# 입력 및 출력 데이터 준비
X = input_sequences
y = tf.keras.utils.to_categorical(output_sequences, num_classes=total_words)

# Seq2Seq 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 64, input_length=max_sequence_length),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.RepeatVector(max_sequence_length),
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(total_words, activation='softmax'))
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X, y, epochs=100, verbose=1)

# 대화 생성 함수
def generate_response(user_input):
    input_seq = tokenizer.texts_to_sequences([user_input])
    padded_input_seq = pad_sequences(input_seq, maxlen=max_sequence_length, padding='post')
    predicted_probabilities = model.predict(padded_input_seq, verbose=0)[0]
    predicted_sequence = [np.argmax(probabilities) for probabilities in predicted_probabilities]
    response = tokenizer.sequences_to_texts([predicted_sequence])[0]
    return response

# 대화 테스트
user_input = "안녕"
response = generate_response(user_input)
print("챗봇:", response)
