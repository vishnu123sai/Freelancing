# chatbot bulding 


# importing required packages
import re
import tensorflow as tf

# loading text files 
lines = open("cornell movie-dialogs corpus/movie_lines.txt", 'r').read().split("\n")
conversations = open("cornell movie-dialogs corpus/movie_conversations.txt", 'r').read().split("\n")


lines_to_conversations = {}
for _line in lines:
    _temp  = _line.split(" +++$+++ ")
    lines_to_conversations[_temp[0]] = _temp[-1]

conversations_lines = []
for _conversation in conversations:
    _temp  = _conversation.split(" +++$+++ ")[-1][1:-1].replace("'","").replace(" ","")
    conversations_lines.append(_temp.split(","))
    
# getting questions and answers seperately
questions = []
answers = []
for _question_answers in conversations_lines:
    for _i in range(len(_question_answers)-1):
        questions.append(lines_to_conversations[_question_answers[_i]])
        answers.append(lines_to_conversations[_question_answers[_i+1]])
 

# cleaning text        
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", " i am", text)
    text = re.sub(r"he's", " he is", text)
    text = re.sub(r"she's", " she is", text)
    text = re.sub(r"that's", " that is", text)
    text = re.sub(r"what's", " what is", text)
    text = re.sub(r"where's", " where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", " will not", text)
    text = re.sub(r"can't", " can not", text)
    text = re.sub(r"[-{}()~!@#$%^&*_+=:<>?,.;|//\\]", "", text)
    return text

# cleaning questions 

clean_questions = []
for _question in questions:
    clean_questions.append(clean_text(_question))


# cleaning answers

clean_answers = []
for _answer in answers:
    clean_answers.append(clean_text(_answer))


# dict with word from question and answer with their occurances
word2count = {}
for _question in clean_questions:
    for _i in _question.split():
        if _i not in word2count:
            word2count[_i]=1
        else:
            word2count[_i] +=1
for _answer in clean_answers:
    for _i in _answer.split():
        if _i not in word2count:
            word2count[_i]=1
        else:
            word2count[_i]+=1
            

# assigning words to unique integers
            
questionwords2int = {}
word_count = 0
threshold  = 20
for word, count in word2count.items():
    if count >= threshold:
        questionwords2int[word] = word_count
        word_count +=1

answerword2int = {}
word_count = 0
for word, count in word2count.items():
    if count >= threshold:
        answerword2int[word] = word_count
        word_count +=1
        
# adding the tokens and giving one unique number to them

tokens = ["<EOS>","<PAD>", "<SOS>","<OUT>"]
for token in tokens:
    questionwords2int[token] = len(questionwords2int)+1
    answerword2int[token] = len(answerword2int)+1
    
# creating inverse answerword2int dictionary to decode 
answerints2word = {w_i : w for w, w_i in answerword2int.items()}

# adding end of string tag to every answer
for i in range(len(clean_answers)):
    clean_answers[i] += "<EOS>"

# converting question to integer list and replacing all the words which are not in word2count
# means occuring less than threshold 

questions_into_ints = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionwords2int:
            ints.append(questionwords2int["<OUT>"])
        else:
            ints.append(questionwords2int[word])
    questions_into_ints.append(ints)

answers_into_ints = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerword2int:
            ints.append(answerword2int["<OUT>"])
        else:
            ints.append(answerword2int[word])
    answers_into_ints.append(ints)


# sorting questions and answers 
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25+1):
    for i in enumerate(questions_into_ints):
        if(len(i[1])==length):
            sorted_clean_questions.append(questions_into_ints[i[0]])
            # making a particular questions to that particular answer
            sorted_clean_answers.append(answers_into_ints[i[0]])



print(word)

# Building Model seq2seq model
            
# placeholder for inputs and targets
# placeholder are decides what type of tensor it is
# tensor is a advance data strcture 

def model_inputs():
    inputs = tf.placeholder(tf.int32, [None,None], name="input" )
    targets = tf.placeholder(tf.int32, [None, None], name="target" )
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    lr = tf.placeholder(tf.float32, name="learning_rate")
    return inputs, targets, keep_prob, lr
 
# keep_prob is the parameter that controls thedropout rate
def preprocessing_target(targets, word2int ,batchsize):
    left_side = tf.fill([batchsize,1], word2int["<SOS>"])
    right_side = tf.strided_slice(targets, [0,0],[batchsize, -1], [1,1])
    preprocessed_target = tf.concat([left_side, right_side],1)
    return preprocessed_target

# creating encoder of RNN Layer
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MinimalRNNCell([lstm_dropout] * num_layers)
    _, encoder_state  = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                        cell_bw = encoder_cell,
                                                        sequence_length = sequence_length,
                                                        inputs = rnn_inputs,
                                                        dtype = tf.float32)
    return encoder_state

# decoding training_set

def decode_training_set(encoder_state, decoder_cell, decoder_embeded_input, sequence_length, decoding_scope,output_function, keep_prob, batchsize):
    attention_state = tf.zeros([batchsize, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_fn, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_state, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_fn,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decode_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                                                             training_decoder_function,
                                                                                                             decoder_embeded_input,
                                                                                                             sequence_length,
                                                                                                             scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
    
#  decoding validation_set

def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, max_length, num_words,sequence_length, decoding_scope,output_function, keep_prob, batchsize):
    attention_state = tf.zeros([batchsize, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_fn, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_state, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    testing_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                                encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_fn,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id, 
                                                                              eos_id, 
                                                                              max_length, 
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decode_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                                                             testing_decoder_function,

                                                                                                             scope = decoding_scope)
   
    return test_predictions

# creating decoder RNN

def decoder_rnn(decoder_embedded_input, decoder_embedding_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batchsize):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MinimalRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x : tf.contrib.layers.fully_connected(x,
                                                                       num_words,
                                                                       None,
                                                                       scope = decoding_scope,
                                                                       weights_initializer = weights,
                                                                       biases_initializer = biases)
        training_predictions =  decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size )
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell, 
                                           decoder_embedding_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length-1, 
                                           num_words, 
                                           decoding_scope,
                                           output_function,
                                           keep_prob, 
                                           batchsize)
    


# building seq2seq model 
def seq2seq_model(inputs, targets, keep_prob, batchsize, sequence_length, answers_num_words, question_num_words, encoder_embedded_size, decoder_embedded_size, rnn_size, num_layers, questionwords2int, ):
    



        



    


    