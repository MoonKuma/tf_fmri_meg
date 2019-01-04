#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : tools.py
# @Author: MoonKuma
# @Date  : 2019/1/4
# @Desc  :



def format_string_para(file_name):
    MAX_WORDS_LINE = 60
    count = 0
    long_str = ''
    with open(file_name,'r',encoding='utf-8') as file_op:
        for line in file_op.read():
            # line = line.strip()
            long_str = long_str+ line
    long_str = long_str.replace('\n',' ')
    words_array = long_str.split(' ')
    return_str = ''
    for word in words_array:
        count = count + len(word) + 1
        return_str = return_str + word + ' '
        if count > MAX_WORDS_LINE:
            return_str += '\n'
            count = 0
    print(return_str)
    return return_str

# test
format_string_para('test.txt')

'''

When trained to predict the next word in a news story, for example, the learned word vectors for Tuesday 
and Wednesday are very similar, as are the word vectors for Sweden and Norway. Such representations are 
called distributed representations because their elements (the features) are not mutually exclusive and 
their many configurations correspond to the variations seen in the observed data. These word vectors 
are composed of learned features that were not determined ahead of time by experts, but automatically 
discovered by the neural network. Vector representations of words learned from text are now very widely 
used in natural language applications. 
'''