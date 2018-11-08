# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:25:44 2018

@author: Ethan Cheng
"""
import helper

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]
view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

import numpy as np
import problem_unittests as tests

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    vocab = set(text)
    vocab_to_int = { w : i for i,w in enumerate(vocab) }
    int_to_vocab = { i : w for i,w in enumerate(vocab) }
    
    return vocab_to_int, int_to_vocab


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    tokens = {
        '.' : '||Period||',
        ',' : '||Comma||',
        '"' : '||Quotation_Mark||',
        ';' : '||Semicolon||',
        '!' : '||Exclamation_mark||',
        '?' : '||Question_mark||',
        '(' : '||Left_Parentheses||',
        ')' : '||Right_Parentheses||',
        '--' : '||Dash||',
        '\n' : '||Return||' }
    
    return tokens

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)

helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)


if 0:
    
    #%%
    import tensorflow as tf
    def get_inputs():
        """
        Create TF Placeholders for input, targets, and learning rate.
        :return: Tuple (input, targets, learning rate)
        """
        # TODO: Implement Function
        Input = tf.placeholder(tf.float32,[None, None], name = "input")
        Targets = tf.placeholder(tf.float32,[None, None], name ="targets")
        LearningRate = tf.placeholder(tf.float32, name = "LR")
        
        return Input, Targets, LearningRate
    
    
    """
    DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
    """
    tests.test_get_inputs(get_inputs)
    
    #%%
    
    def get_init_cell(batch_size, rnn_size):
        """
        Create an RNN Cell and initialize it.
        :param batch_size: Size of batches
        :param rnn_size: Size of RNNs
        :return: Tuple (cell, initialize state)
        """
        # TODO: Implement Function
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        cell = tf.contrib.rnn.MultiRNNCell([lstm] * 2)
        initial_state = tf.identity(cell.zero_state(batch_size, tf.float32),name="initial_state")
        
        return cell, initial_state
    
    
    """
    DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
    """
    tests.test_get_init_cell(get_init_cell)
    
    
    #%%
    
    
    def get_batches(int_text, batch_size, seq_length):
        """
        Return batches of input and target
        :param int_text: Text with the words replaced by their ids
        :param batch_size: The size of batch
        :param seq_length: The length of sequence
        :return: Batches as a Numpy array
        """
        size_per_batch = seq_length * batch_size
        n_batches = len(int_text)//size_per_batch
        len_exact = n_batches * size_per_batch
        int_text = np.array(int_text[:len_exact])
        xx = np.swapaxes(np.reshape(int_text,(batch_size,-1,seq_length)),0,1)
        yy = np.swapaxes(np.reshape(np.roll(int_text,-1),(batch_size,-1,seq_length)),0,1)
    
        batches = []
        for x, y in zip(xx,yy):
            batches.append([x,y])
        return np.array(batches)
    
    
    
    """
    DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
    """
    tests.test_get_batches(get_batches)